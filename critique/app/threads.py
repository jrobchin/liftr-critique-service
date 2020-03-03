import threading


class CallbackThread(threading.Thread):
    """
    Thread with a callback.

    Usage:
        def cb():
            print('callback')

        thread = BaseThread(
            name='test',
            target=my_thread_job,
            callback=(cb, ("hello", "world")),
        )

        thread.start()
    """

    def __init__(self, callback:tuple=None, *args, **kwargs):
        target = kwargs.pop('target')
        super().__init__(target=self.target_with_callback, *args, **kwargs)
        self.method = target
        
        self.callback = callback[0]
        if len(callback) > 1:
            self.callback_args = callback[1]
        else:
            self.callback_args = ()

    def target_with_callback(self):
        self.method()
        if self.callback is not None:
            self.callback(*self.callback_args)