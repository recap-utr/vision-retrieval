from ft_simclr_pl_treemap_sat import main

datasets = [
    ("../saved_models/PretrainAllNew/dot_layout.ckpt", "kblw/graphimages_dot"), 
            ("../saved_models/PretrainAllNew/new/twopi_layout_v2.ckpt", 
            "kblw/graphimages_twopi"),
              ("../saved_models/PretrainAllNew/new/treemap_weak_v3.ckpt", "kblw/treemap_weak_ft"), 
              ("../saved_models/PretrainAllNew/new/treemap_sat_v4.ckpt", "kblw/treemap_sat_ft")
            ]

if __name__ == "__main__":
    for entry in datasets:
        if len(entry) == 3:
            main(entry[0], entry[1], revision=entry[2])
        else:
            main(entry[0], entry[1])