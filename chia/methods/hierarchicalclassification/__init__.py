from chia.methods.hierarchicalclassification import keras_idk_hc, keras_onehot_hc

_method_mapping = {
    "keras::OneHot": keras_onehot_hc.OneHotEmbeddingBasedKerasHC,
    "keras::IDK": keras_idk_hc.IDKEmbeddingBasedKerasHC,
}


def methods():
    return _method_mapping.keys()


def method(key, kb):
    return _method_mapping[key](kb)
