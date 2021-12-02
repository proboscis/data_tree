from data_tree.auto_data.custom_rules import custom_rule_book
from data_tree.auto_data.rules import RecursiveRuleBook, AutoRuleBook
from data_tree.coconut.auto_data import AutoSolver
from data_tree.coconut.convert import AutoImage, ImageDef, PILImage
from data_tree.coconut.omni_converter import cast_imdef_to_dict, cast_imdef_str_to_imdef, cast_imdef_to_imdef_str, \
    dict2imdef, cast_ary_str_to_ary_type, img_list_is_imgs, lll_is_rgb, cast_tuple2auto_tuple, any2widget, rule_format, \
    rule_imgs2tile, rule_batch_to_tiled, rule_img2widget, rule_image2gray, rule_image_path_to_image, \
    intra_list_conversions, numpys_to_numpy, tensor_to_list, pil_convert, rgb_to_rgba, repeat_ch, \
    torch_img_to_pixpix_input, torch_img_to_vgg_prep, rule_to_spectrum, auto_tuple2widget, rule_numpy2img
from data_tree.coconut.rules.to_bytes import rule_image_str_to_jpg_bytes
from loguru import logger

DEFAULT_RULES = AutoRuleBook(
    AutoImage.default_rules.copy() + [
        AutoSolver.create_cast_rule(cast_imdef_to_dict, "cast_imdef_to_dict"),
        AutoSolver.create_cast_rule(cast_imdef_str_to_imdef, "cast_imdef_str_to_imdef"),
        AutoSolver.create_cast_rule(cast_imdef_to_imdef_str, "cast_imdef_to_imdef_str"),
        AutoSolver.create_cast_rule(dict2imdef, "dict2imdef"),
        AutoSolver.create_cast_rule(cast_ary_str_to_ary_type, "cast_ary_str_to_ary_type"),
        AutoSolver.create_cast_rule(img_list_is_imgs, "img_list_is_imgs"),
        AutoSolver.create_cast_rule(lll_is_rgb, "lll_is_rgb", cost=10),
        AutoSolver.create_cast_rule(cast_tuple2auto_tuple, "tuple <--> auto_tuple"),
        # AutoSolver.create_cast_rule(dict2visdomable),
        AutoSolver.create_conversion_rule(any2widget),
        AutoSolver.create_conversion_rule(rule_format),
        AutoSolver.create_conversion_rule(rule_imgs2tile),
        AutoSolver.create_conversion_rule(rule_batch_to_tiled),
        AutoSolver.create_conversion_rule(rule_img2widget),
        AutoSolver.create_conversion_rule(rule_numpy2img),
        AutoSolver.create_conversion_rule(rule_image2gray),
        AutoSolver.create_conversion_rule(rule_image_path_to_image),
        AutoSolver.create_conversion_rule(rule_image_str_to_jpg_bytes),
        # AutoSolver.create_conversion_rule(rule_image2lab),
        # AutoSolver.create_conversion_rule(rule_rgba2laba),
        # AutoSolver.create_conversion_rule(rule_lab_value_conversion),
        AutoSolver.create_conversion_rule(intra_list_conversions),
        AutoSolver.create_conversion_rule(numpys_to_numpy),
        AutoSolver.create_conversion_rule(tensor_to_list),
        AutoSolver.create_conversion_rule(pil_convert),
        AutoSolver.create_conversion_rule(rgb_to_rgba),
        AutoSolver.create_conversion_rule(repeat_ch),
        AutoSolver.create_conversion_rule(torch_img_to_pixpix_input),
        AutoSolver.create_conversion_rule(torch_img_to_vgg_prep),
        # AutoSolver.create_conversion_rule(intra_tuple_conversions),
        AutoSolver.create_conversion_rule(rule_to_spectrum),
        AutoSolver.create_conversion_rule(auto_tuple2widget),
        AutoSolver.create_alias_rule("numpy_rgb", "numpy,uint8,HWC,RGB,0_255"),
        AutoSolver.create_alias_rule("numpy_rgba", "numpy,uint8,HWC,RGBA,0_255"),
    ]) + custom_rule_book

try:
    import wandb
    def img_to_wandb_img(state):
        from pampy import _
        if isinstance(state, ImageDef) and isinstance(state.data_type, PILImage):
            return [(
                lambda img: wandb.Image(img),
                "wandb.Image",
                "image to wandb image",
                1
            )]

    DEFAULT_RULES = DEFAULT_RULES.add_rule(img_to_wandb_img)
    logger.warning(f"added wandb related conversions")
except Exception as e:
    logger.warning(f"could not add wandb related conversions since wandb could not be imported")

DEFAULT_RULES = RecursiveRuleBook(DEFAULT_RULES)


# now I can add as many recursive rules..
# THE ANSWER: RE_IMPLEMENT the AstarSolver with Design to support providing a solver or solving dependency while searching!
# change intra_list_conversion to use a given solver or neighbors
# so that dict of anything can be converted into {any_state} form when asked.
# so {undefined} => {widget} => widget
# but {widget} cannot be automatically found from current search
# ok so just ask for {undefined} and turn it into widget.
# how do you distinguish it from imdef type explained in dict?
# put _type field in the numpy dict and then ignore them.
#


