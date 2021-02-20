from AttributeDataset import AttributeGameDataset

n_attributes, attribute_size = 3,4
data = AttributeGameDataset(n_attributes,attribute_size,n_remove_classes=4,train=True)