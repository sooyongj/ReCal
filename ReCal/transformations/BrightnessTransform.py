import torchvision.transforms.functional as TF


class BrightnessTransform:

  def __init__(self, brightness):
    assert brightness > 0.0
    self.brightness = brightness

  def __call__(self, x):
    return TF.adjust_brightness(x, self.brightness)