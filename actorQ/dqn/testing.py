

class simple_dataitem:
  def __init__(self, samples):
      self.data = samples

class simple_dataset:
  def __init__(self):
      pass

  def __iter__(self):
    return self

  def __next__(self):
    while True:
        return simple_dataitem(1)

x = simple_dataset()
x = iter(x)
print(next(x).data)
