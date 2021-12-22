import torchvision.transforms.functional as TF


def brightness(img, factor):
  img = TF.adjust_brightness(img, factor)
  return img


def zoom_out(img, scale):
  def compute_zoom_pixel(width, scale_factor):
    added_pad = int(width * (1 - scale_factor) / scale_factor)
    if added_pad % 2 == 0:
      return added_pad // 2
    else:
      return added_pad // 2, added_pad // 2, added_pad // 2 + 1, added_pad // 2 + 1
  origin_px = img.shape[-1]
  pad = compute_zoom_pixel(origin_px, scale)
  img = TF.pad(img, pad)
  img = TF.resize(img, origin_px)
  return img


def get_normalization_func(params):
  if params is None:
    return None

  assert len(params) == 0

  return lambda img: TF.normalize(img, params[0], params[1])
