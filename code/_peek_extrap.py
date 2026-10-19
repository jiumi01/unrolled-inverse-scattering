# -*- coding: utf-8 -*-
import json, numpy as np
d = json.load(open('../results/extrap.json'))
for k, v in d.items():
    a = np.array(v['rre']); r = np.array(v['residual'])
    print(k, 'rre len', len(a))
    print('   rre head', np.round(a[:3], 4), 'tail', np.round(a[-3:], 4))
    print('   res head', r[:3].round(4), 'tail', r[-3:].round(4))
