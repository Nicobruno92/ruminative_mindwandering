class BehNotFound(Exception):
    def __init__(self, message="Beheavioral data not found"):
        self.message = message
        super().__init__(self.message)

class InconsistentAnnotationsWithBeh(Exception):
    def __init__(self, message="Annotations are inconsistent with behavioral data"):
        self.message = message
        super().__init__(self.message)

