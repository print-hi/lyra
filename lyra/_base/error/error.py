class AttributeDoesNotExist(Exception):
    def __init__(self, call):
        self.err = "Attribute is created after calling <Classifier>" + call
        super().__init__(self.err)

    def __str__(self):
        return f'{self.err}'