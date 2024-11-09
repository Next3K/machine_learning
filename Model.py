class Model:
    class_attribute = "some value"

    def __init__(self, k: int, mask: [bool]):
        self.k = k
        self.mask = mask

    def method_name(self):
        return f"Attributes: {self.k} and {self.mask}"
     
