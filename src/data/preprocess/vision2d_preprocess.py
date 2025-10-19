class Vision2DPreprocess:
    def __init__(self, vision2d_processor, data_arguments=None):
        self.vision2d_processor = vision2d_processor

    def __call__(self, vision2d):
        vision2d = self.vision2d_processor(vision2d, return_tensors="pt")
        vision2d = vision2d["pixel_values"][0]
        # vision2d = self.vision2d_processor(vision2d)
        return vision2d
