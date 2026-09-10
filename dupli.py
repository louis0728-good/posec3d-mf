import pickle, re
d = pickle.load(open('pkl/train_val.pkl','rb'))
tr, va = d['split']['xsub_train'], d['split']['xsub_val']
cid = lambda n: re.sub(r'_p\d+$', '', n)      # 依你的命名規則調整

print('annotations =', len(d['annotations']), ' train+val =', len(tr)+len(va))
print('🔴 train ∩ val         =', len(set(tr) & set(va)))
print('🔴 train 內重複         =', len(tr) - len(set(tr)))
print('🔴 clip 級重疊 (p1/p2)  =', len({cid(n) for n in tr} & {cid(n) for n in va}))