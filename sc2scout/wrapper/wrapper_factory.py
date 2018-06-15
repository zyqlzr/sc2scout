
class WrapperMaker(object):
    def __init__(self, name):
        self.name = name

    def make_wrapper(self, env):
        raise NotImplementedError


class MakerFactory(object):
    def __init__(self):
        self.makers = {}

    def register(self, name, maker):
        print('register wrapper_maker ', name)
        if name in self.makers:
            return False
        self.makers[name] = maker
        return True

    def make(self, name, env):
        if name in self.makers:
            maker = self.makers[name]
            return maker.make_wrapper(env)
        else:
            print('can not find wrapper maker ', name)
            return None

GlobalMakerFactory = MakerFactory()

def register(name, maker):
    return GlobalMakerFactory.register(name, maker)

def make(name, env):
    return GlobalMakerFactory.make(name, env)

