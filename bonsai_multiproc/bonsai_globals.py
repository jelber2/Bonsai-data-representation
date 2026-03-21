import sys
import threading

globalVars = {'nCells', 'nGenes', 'nNodes', 'max_node_ind', 'mem_friendly'}

_THREAD_LOCAL_ATTRS = frozenset({'max_node_ind', 'nNodes'})

class _BonsaiGlobals:
    """Module-replacement that stores max_node_ind and nNodes as thread-local.

    Every thread (main or worker) gets its own copy of max_node_ind and nNodes.
    All other attributes (nCells, nGenes, mem_friendly) are shared as usual.

    Initial values for worker threads are inherited from the main thread's last
    write via _shared_defaults, which is updated only when the main thread writes.
    """

    globalVars = globalVars
    _tls = threading.local()
    _shared_defaults = {
        'nCells': None,
        'nGenes': None,
        'nNodes': None,
        'max_node_ind': 0,
        'mem_friendly': False,
    }
    _shared_non_tls = {
        'nCells': None,
        'nGenes': None,
        'mem_friendly': False,
    }

    def __getattr__(self, name):
        if name in _THREAD_LOCAL_ATTRS:
            val = getattr(self._tls, name, None)
            if val is None:
                val = self._shared_defaults.get(name)
            return val
        if name in self._shared_non_tls:
            return self._shared_non_tls[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'globalVars':
            object.__setattr__(self, name, value)
            return
        if name in _THREAD_LOCAL_ATTRS:
            setattr(self._tls, name, value)
            if threading.current_thread() is threading.main_thread():
                self._shared_defaults[name] = value
        else:
            self._shared_non_tls[name] = value


sys.modules[__name__] = _BonsaiGlobals()
