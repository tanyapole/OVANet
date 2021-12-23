import os
import torch


def log_set(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    network = kwargs["network"]
    conf_file = kwargs["config_file"]
    script_name = kwargs["script_name"]
    multi = kwargs["multi"]
    #args = kwargs["args"]

    target_data = os.path.splitext(os.path.basename(target_data))[0]
    logname = "{file}_{source}2{target}_{network}_hp_{hp}".format(file=script_name.replace(".py", ""),
                                                                               source=source_data.split("_")[1],
                                                                               target=target_data,
                                                                               network=network,
                                                                               hp=str(multi))
    logname = os.path.join("record", kwargs["exp_name"],
                           os.path.basename(conf_file).replace(".yaml", ""), logname)
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname




def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c
