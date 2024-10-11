# create a class for problem constants to easier pass those around .py files
class problem_constants:
    def __init__(self, width, height, channel_count, categories_count, batch_size):
        self.width = width
        self.height = height
        self.channel_count = channel_count
        self.categories_count = categories_count
        self.batch_size = batch_size