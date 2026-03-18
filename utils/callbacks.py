# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """"
    处理所有注册的回调函数，供钩子使用
    """

    # 定义可用的回调函数钩子
    _callbacks = {
        'on_pretrain_routine_start': [],  # 预训练过程开始前
        'on_pretrain_routine_end': [],    # 预训练过程结束后

        'on_train_start': [],             # 训练开始
        'on_train_epoch_start': [],       # 训练周期开始
        'on_train_batch_start': [],       # 训练批次开始
        'optimizer_step': [],             # 优化器步进
        'on_before_zero_grad': [],        # 在清零梯度之前
        'on_train_batch_end': [],         # 训练批次结束
        'on_train_epoch_end': [],         # 训练周期结束

        'on_val_start': [],               # 验证开始
        'on_val_batch_start': [],         # 验证批次开始
        'on_val_image_end': [],           # 验证图像处理结束
        'on_val_batch_end': [],           # 验证批次结束
        'on_val_end': [],                 # 验证结束

        'on_fit_epoch_end': [],           # 适配周期结束（包括训练和验证）
        'on_model_save': [],              # 模型保存
        'on_train_end': [],               # 训练结束

        'teardown': [],                   # 清理工作
    }

    def register_action(self, hook, name='', callback=None):
        """
        将一个新的动作注册到指定的回调钩子

        参数:
            hook        要注册动作的回调钩子名称
            name        动作的名称，供以后引用
            callback    要触发的回调函数
        """
        # 确保指定的钩子存在于回调函数字典中
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        # 确保提供的回调函数是可调用的
        assert callable(callback), f"callback '{callback}' is not callable"
        # 将新的动作添加到指定钩子的回调函数列表中
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """"
        返回所有注册的动作，按回调钩子分类

        参数:
            hook 要检查的钩子名称，默认为所有
        """
        if hook:
            # 返回指定钩子的所有注册动作
            return self._callbacks[hook]
        else:
            # 返回所有钩子的所有注册动作
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        遍历注册的动作并触发所有回调函数

        参数:
            hook  要检查的钩子名称
            args  传递给回调函数的位置参数
            kwargs  传递给回调函数的关键字参数
        """

        # 确保指定的钩子存在于回调函数字典中
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        # 遍历所有注册在指定钩子下的回调函数，并触发它们
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)
