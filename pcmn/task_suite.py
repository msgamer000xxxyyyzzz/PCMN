"""Deterministic synthetic reasoning dataset with exact verifiers.

Generated examples are suitable for testing training mechanics, but their
success does not establish broad reasoning, causality, or coding ability.
"""
from __future__ import annotations
import ast, json, random
from pathlib import Path


def make_items(n=100,seed=42):
    rng=random.Random(seed)
    items=[]
    for i in range(n):
        a=rng.randrange(1,1000); b=rng.randrange(1,1000); c=rng.randrange(1,50)
        if i%4==0:
            text=f'Вычисли точно: ({a} + {b}) * {c}.'
            ans=str((a+b)*c)
            category='arithmetic'
        elif i%4==1:
            # Structural causal model with an explicit intervention.
            # Z = 2*X + 3*Y; do(X=a) replaces X, not Y.
            text=f'Причинная модель: Z = 2*X + 3*Y. При do(X={a}), Y={c}. Найди Z.'
            ans=str(2*a+3*c)
            category='causal_do'
        elif i%4==2:
            text=f'Переменные: x={a}, y={b}. Затем x=x+y, y=x-y. Чему равно y?'
            ans=str(a)
            category='state_tracking'
        else:
            text=f'Напиши Python-функцию sum_{i}(x), которая возвращает x + {c}.'
            ans=f'def sum_{i}(x):\n    return x + {c}'
            category='code_generation'
        items.append({'category':category,'question':text,'answer':ans})
    return items


def write_chat(items,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps({'messages':[{'role':'user','content':item['question']},
                                             {'role':'assistant','content':item['answer']}],
                                'category':item['category']},ensure_ascii=False)+'\n')


def check_code(text,func_name,arg,expected):
    """Safely verify AST structure, never execute untrusted model code."""
    try:tree=ast.parse(text)
    except SyntaxError:return False
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
    fun=tree.body[0]
    if fun.name != func_name or len(fun.args.args)!=1 or len(fun.body)!=1:return False
    ret=fun.body[0]
    if not isinstance(ret,ast.Return) or not isinstance(ret.value,ast.BinOp) or not isinstance(ret.value.op,ast.Add):return False
    l,r=ret.value.left,ret.value.right
    return (isinstance(l,ast.Name) and l.id==fun.args.args[0].arg and
            isinstance(r,ast.Constant) and type(r.value)==int and arg+r.value==expected)
