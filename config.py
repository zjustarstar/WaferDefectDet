
# 返回结果
RESULT_OK = 0
RESULT_FAIL = -1
RESULT_FAIL_NO_MATCHBLOCK = -2    # 找不到匹配的区域
RESULT_FAIL_WRONGPARAM = -3       # 不合理的参数

# 缺陷检测结果
AD_GOOD = False
AD_NG = True

# 一些参数的配置
LOG_NAME = "WDD_LOG"
COMMAND_ID = "command_id"         # 表示命令编号的字符串
SHARE_DIR = "e:\\camera_data\\"   # 共享文件夹路径
WCF_SERVER_ADDRESS = 'http://localhost:8733//WCFEquipControlService?singleWsdl'   #wcf服务地址;

#算法阈值
ALG_MATCH_THRESHOLD = 0.4        # 模板匹配的阈值
