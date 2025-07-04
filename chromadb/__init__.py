class Collection:
    def __init__(self):
        pass
    def add(self, documents=None, metadatas=None, ids=None):
        pass
    def update(self, ids=None, metadatas=None):
        pass

class PersistentClient:
    def __init__(self, path):
        self.path = path
        self.collection = Collection()
    def get_or_create_collection(self, name):
        return self.collection
