from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.gtot_path = '/data/GTOT/'
    settings.lasher_path = '/data/LasHeR/'
    settings.lasot_extension_subset_path = ''
    settings.lasot_path = ''
    settings.network_path = '/data/wangwanyu/Codes/AFter/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/data/OTB100'
    settings.oxuva_path = ''
    settings.result_plot_path = '/data/wangwanyu/Codes/AFter/pytracking/result_plots/'
    settings.results_path = '/data/wangwanyu/Codes/AFter/pytracking/tracking_results/'    # Where to store tracking results
    settings.rgbt210_path = '/data/RGBT210/'
    settings.rgbt234_path = '/data/RGBT234/'
    settings.segmentation_path = '/data/wangwanyu/Codes/AFter/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.vtuavlt_path = '/data/VTUAV/test_LT/'
    settings.vtuavst_path = '/data/VTUAV/test_ST/'
    settings.youtubevos_dir = ''

    return settings

