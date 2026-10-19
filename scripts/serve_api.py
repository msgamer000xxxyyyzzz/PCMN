"""Local text inference API, partial OpenAI-style request/response schema.

Default listens ONLY on 127.0.0.1; remote bind requires PCMN_API_KEY.
No TLS or rate limiting: put behind trusted authenticated reverse proxy.
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,os,time,uuid
from pcmn.serving import TextEngine
from pcmn.agent import render_prompt


def handler_factory(engine,key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            # Avoid logging request body/prompt, but show sanitized base request lines.
            return

        def reply(self,status,payload):
            data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)

        def authorized(self):
            return not key or self.headers.get('Authorization')==f'Bearer {key}'

        def do_GET(self):
            if not self.authorized():return self.reply(401,{'error':'Unauthorized'})
            if self.path=='/health':return self.reply(200,{'status':'ok','model':engine.model_name})
            if self.path=='/v1/models':
                return self.reply(200,{'object':'list','data':[{'id':engine.model_name,'object':'model'}]})
            return self.reply(404,{'error':'Not found'})

        def do_POST(self):
            if not self.authorized():return self.reply(401,{'error':'Unauthorized'})
            if self.path not in ('/v1/chat/completions','/v1/completions'):
                return self.reply(404,{'error':'Not found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 1<=size<=262144:raise ValueError('Body must be 1-262144 bytes')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Expected JSON object')
                is_chat=self.path=='/v1/chat/completions'
                if is_chat:
                    messages=data['messages']
                    if not isinstance(messages,list) or not 1<=len(messages)<=64:
                        raise ValueError('Invalid messages')
                    if any(not isinstance(m,dict) or m.get('role') not in {'system','user','assistant','tool'}
                           or not isinstance(m.get('content'),str) for m in messages):
                        raise ValueError('Invalid message role/content')
                    prompt=render_prompt(messages)
                else:
                    prompt=data['prompt']
                    if not isinstance(prompt,str):raise ValueError('Prompt must be string')
                kw={'max_tokens':int(data.get('max_tokens',128)),
                    'temperature':float(data.get('temperature',0.8)),
                    'top_k':int(data.get('top_k',40)),
                    'greedy':bool(data.get('greedy',False))}
                stream=data.get('stream',False)
                if type(stream) is not bool:raise ValueError('stream must be boolean')
                if not 1<=kw['max_tokens']<=2048 or not 0<kw['temperature']<=5:
                    raise ValueError('Invalid generation limits')
                if not stream:
                    result=engine.complete(prompt,**kw)
                    content={'role':'assistant','content':result['text']} if is_chat else result['text']
                    self.reply(200,{'id':'pcmn-'+uuid.uuid4().hex,'object':
                        'chat.completion' if is_chat else 'text_completion',
                        'created':int(time.time()),'model':engine.model_name,
                        'choices':[{'index':0,'message':content,'text':result['text'],
                                    'finish_reason':'stop' if result['tokens']<kw['max_tokens'] else 'length'}],
                        'usage':{'prompt_tokens':result['prompt_tokens'],
                                  'completion_tokens':result['tokens'],
                                  'total_tokens':result['prompt_tokens']+result['tokens']}})
                else:
                    # Progressive token-by-token SSE. Byte UTF-8 sequences are
                    # buffered; BPE may buffer text until a valid prefix emerges.
                    import codecs
                    decoder=codecs.getincrementaldecoder('utf-8')('replace')
                    self.send_response(200)
                    self.send_header('Content-Type','text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control','no-cache')
                    self.end_headers()
                    acc=[];previous=''
                    for token in engine.tokens(prompt,**kw):
                        acc.append(token)
                        if engine.tokenizer.kind=='byte':
                            piece=decoder.decode(bytes([token-3])) if 3<=token<=258 else ''
                        else:
                            now=engine.tokenizer.decode(acc)
                            piece=now[len(previous):] if now.startswith(previous) else ''
                            previous=now
                        if not piece:continue
                        delta={'role':'assistant','content':piece} if is_chat else piece
                        chunk={'object':'chat.completion.chunk' if is_chat else 'text_completion',
                               'model':engine.model_name,
                               'choices':[{'index':0,'delta':delta,'text':piece,'finish_reason':None}]}
                        self.wfile.write(('data: '+json.dumps(chunk,ensure_ascii=False)+'\n\n').encode('utf-8'))
                        self.wfile.flush()
                    if engine.tokenizer.kind=='byte':
                        tail=decoder.decode(b'',final=True)
                        if tail:
                            delta={'role':'assistant','content':tail} if is_chat else tail
                            self.wfile.write(('data: '+json.dumps({'choices':[{'index':0,'delta':delta}]},ensure_ascii=False)+'\n\n').encode())
                    self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            except (ValueError,KeyError,TypeError,OverflowError,RuntimeError) as exc:
                # Do not expose tracebacks. A failed stream cannot send JSON.
                try:
                    if not self.wfile.closed and not self.headers.get('Content-Type')=='text/event-stream':
                        self.reply(400,{'error':str(exc)[:250]})
                except (OSError,ConnectionError):pass
            except (BrokenPipeError,ConnectionResetError):pass
    return Handler


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--device',default='cpu')
    p.add_argument('--devices',help='Comma-separated one device per model layer, e.g. cuda:0,cuda:1,cpu')
    p.add_argument('--cpu-lib')
    p.add_argument('--num-threads',type=int,default=2)
    p.add_argument('--memory-index',help='Optional UTF-8 JSONL passage index')
    p.add_argument('--retrieval-k',type=int,default=4)
    p.add_argument('--trace-cpu',action='store_true')
    p.add_argument('--grounded',action='store_true',help='Use JSON-quoted retrieval evidence with source IDs')
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.num_threads)
    key=os.environ.get('PCMN_API_KEY','')
    if a.host not in ('127.0.0.1','localhost','::1') and not key:
        raise SystemExit('Remote bind requires PCMN_API_KEY; use TLS proxy')
    engine=TextEngine(a.checkpoint,device=a.device,
                      devices=a.devices.split(',') if a.devices else None,cpu_lib=a.cpu_lib,
                      memory_index=a.memory_index,retrieval_k=a.retrieval_k,trace_cpu=a.trace_cpu,grounded=a.grounded)
    server=ThreadingHTTPServer((a.host,a.port),handler_factory(engine,key))
    print(f'PCMN local inference: http://{a.host}:{a.port} ; model={engine.model_name}',flush=True)
    server.serve_forever()
if __name__=='__main__':main()
