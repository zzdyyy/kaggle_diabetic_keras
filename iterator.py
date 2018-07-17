import multiprocessing
import threading
import queue
from uuid import uuid4
import numpy as np

import data


class BatchIterator(object):
    """when given a dataset X and y, generate batch, transform it and yield the batch"""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None, transform=None, color_vec=None):
        self.tf = transform
        self.color_vec = color_vec
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


class QueueIterator(BatchIterator):
    """BatchIterator with seperate thread to do the image reading."""

    def __iter__(self):
        myqueue = queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for Xb, yb in super(QueueIterator, self).__iter__():
                #print("I'm putting one batch in the queue...")
                myqueue.put((np.array(Xb), np.array(yb)))
            myqueue.put(end_marker)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = myqueue.get()
        while item is not end_marker:
            yield item
            myqueue.task_done()
            item = myqueue.get()


multi_reading = False
if multi_reading:
    import SharedArray  # see SharedIterator below

    def load_shared(args):
        i, array_name, fname, kwargs = args
        array = SharedArray.attach(array_name)
        array[i] = data.load_augment(fname, **kwargs)

    global_pool = multiprocessing.Pool(4)

    class SharedIterator(QueueIterator):
        """override the transform function to read image

        This class use SharedArray to do multi-process image image read and augment. The global_pool seems work better
        when initialized early (but after the load_shared function).

        Sometimes the multi-process may be conflict with tensorflow/keras and cause bus error. If this happens, try to
        set multi_reading = False

        If the running process exits by exception (e.g. Ctrl+C), there will be some trash leaved in /dev/shm, which can
        be deleted manually, or automatically when the system reboots.
        """
        def __init__(self, config, deterministic=False, *args, **kwargs):
            self.config = config
            self.deterministic = deterministic
            self.pool = global_pool
            super(SharedIterator, self).__init__(*args, **kwargs)

        def transform(self, Xb, yb):

            shared_array_name = str(uuid4())
            try:
                shared_array = SharedArray.create(
                    shared_array_name, [len(Xb), 3, self.config.get('w'),
                                        self.config.get('h')], dtype=np.float32)

                fnames, labels = super(SharedIterator, self).transform(Xb, yb)
                args = []

                for i, fname in enumerate(fnames):
                    kwargs = {k: self.config.get(k) for k in ['w', 'h']}
                    if not self.deterministic:
                        kwargs.update({k: self.config.get(k)
                                       for k in ['aug_params', 'sigma']})
                    kwargs['transform'] = getattr(self, 'tf', None)
                    kwargs['color_vec'] = getattr(self, 'color_vec', None)
                    args.append((i, shared_array_name, fname, kwargs))

                self.pool.map(load_shared, args)
                Xb = np.array(shared_array, dtype=np.float32)

            finally:
                SharedArray.delete(shared_array_name)

            if labels is not None:
                labels = labels[:, np.newaxis]

            return Xb, labels

else:  # if multi_reading == False
    class SharedIterator(QueueIterator):
        """overwrite the transform function to read image

        Single-process implementation of SharedIterator
        """

        def __init__(self, config, deterministic=False, *args, **kwargs):
            self.config = config
            self.deterministic = deterministic
            super(SharedIterator, self).__init__(*args, **kwargs)

        def transform(self, Xb, yb):

            fnames, labels = super(SharedIterator, self).transform(Xb, yb)

            array = []
            for i, fname in enumerate(fnames):
                kwargs = {k: self.config.get(k) for k in ['w', 'h']}
                if not self.deterministic:
                    kwargs.update({k: self.config.get(k)
                                   for k in ['aug_params', 'sigma']})
                kwargs['transform'] = getattr(self, 'tf', None)
                kwargs['color_vec'] = getattr(self, 'color_vec', None)
                array.append(data.load_augment(fname, **kwargs))

            Xb = np.stack(array).astype('float32')

            if labels is not None:
                labels = labels[:, np.newaxis]

            return Xb, labels


class ResampleIterator(SharedIterator):
    def __init__(self, config, initial_epoch=0, *args, **kwargs):
        self.config = config
        self.count = initial_epoch
        super(ResampleIterator, self).__init__(config, *args, **kwargs)

    def __call__(self, X, y=None, transform=None, color_vec=None):
        if y is not None:
            alpha = self.config.cnf['balance_ratio'] ** self.count
            class_weights = self.config.cnf['balance_weights'] * alpha \
                + self.config.cnf['final_balance_weights'] * (1 - alpha)
            self.count += 1
            indices = data.balance_per_class_indices(y, weights=class_weights)
            X = X[indices]
            y = y[indices]
        return super(ResampleIterator, self).__call__(X, y, transform=transform,
                                                      color_vec=color_vec)

