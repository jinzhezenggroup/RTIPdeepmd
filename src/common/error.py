from __future__ import annotations

from .constants import AminoAcid, Element, FragmentType


def error_file(operation: str, filename: str) -> str:
    return f"\n\n\n ERROR: There is some problem in {operation} the file '{filename}'. \n\n\n"


def error_read(filename: str) -> str:
    return f"\n\n\n ERROR: There is some problem with the input file '{filename}'. Please check it. \n\n\n"


def error_dir(operation: str, dir_name: str) -> str:
    return (
        f"\n\n\n ERROR: There is some problem in {operation} the directory '{dir_name}'. "
        "Maybe it already exists or you have no permission. \n\n\n"
    )


def error_type(variable: str, illegal_type: str) -> str:
    return f"\n\n\n ERROR: Illegal '{variable}' type '{illegal_type}' has read from the input file. Please check it. \n\n\n"


def error_fragment_format(fragment_type: FragmentType, filename: str) -> str:
    return f"\n\n\n ERROR: Fragment {fragment_type} in the input file '{filename}' has an illegal format. Please check it. \n\n\n"


def error_static_hashmap(key: str, value: str, hashmap: str) -> str:
    return (
        f"\n\n\n ERROR: There is some problem in getting the value '{value}' by key '{key}' "
        f"in the static HashMap '{hashmap}'. Please check it. \n\n\n"
    )


def error_getting_property(property_name: str, object_name: str) -> str:
    return f"\n\n\n ERROR: There is some problem in getting the {property_name} of {object_name}. Please check it. \n\n\n"


def error_str_to_cstring(str_name: str) -> str:
    return f"\n\n\n ERROR: There is some problem in transforming str '{str_name}' to CString. \n\n\n"


def error_type_transformation(variable: str, type1: str, type2: str) -> str:
    return f"\n\n\n ERROR: There is some problem in type transformation of '{variable}' from {type1} to {type2}. \n\n\n"


def error_as_slice(variable: str) -> str:
    return f"\n\n\n ERROR: There is some problem in getting the slice of the variable '{variable}'. \n\n\n"


def error_none_value(variable: str) -> str:
    return f"\n\n\n ERROR: There is some problem with variable '{variable}', which has none/wrong value. \n\n\n"


def error_cloning(variable: str) -> str:
    return f"\n\n\n ERROR: There is some problem in cloning the variable '{variable}'. \n\n\n"


def error_non_bioelement_nn(element: Element) -> str:
    return f"\n\n\n ERROR: The neural network of Element '{element}' has't been established yet, since it's not the bioelement. Please check it. \n\n\n"


def error_intrafragment_descriptor(amino_acid: AminoAcid, n: int) -> str:
    return f"\n\n\n ERROR: The intrafragment descriptor of Amino_Acid '{amino_acid}' should have '{n}' values. Please check it. \n\n\n"


def error_global_data_size() -> str:
    return "\n\n\n ERROR: In NN training, there should have at least two data for a specific protein system, one for training set, the other for validation set. Please check it. \n\n\n"


def error_nn_para_update(fragment: str, optimizer: str) -> str:
    return f"\n\n\n ERROR: There is some problem in updating the parameters of sub-NN of '{fragment}' using the optimizer '{optimizer}'. Please check it. \n\n\n"


def error_min_1d() -> str:
    return "\n\n\n ERROR: There is some problem with the function min_1d: the input fun is increasing along +x direction, or the default minimum step is too large. \n\n\n"
