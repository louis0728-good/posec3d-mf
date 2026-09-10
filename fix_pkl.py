import sys
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric
import numpy.core._multiarray_umath
import numpy.core.umath

# 把 numpy 2.x 的 numpy._core 對應到 numpy 1.x 的 numpy.core
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric
sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
sys.modules['numpy._core.umath'] = numpy.core.umath

import mmengine

src = 'pkl/train_val.pkl'          # 壞掉的(numpy 2.x 存的)
dst = 'pkl/train_val_fixed.pkl'    # 修好的(numpy 1.x 格式)

data = mmengine.load(src)          # 這裡就不會再噴 numpy._core 了
mmengine.dump(data, dst)
print(f'修好了 -> {dst}')
print('樣本數:', len(data['annotations']))