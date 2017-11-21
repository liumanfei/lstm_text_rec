# -*- coding: utf-8 -*-  
import os
import numpy as np
import datetime
import random
import math
import heapq
import itertools
# import struct
# import pylab
# # from PIL import Image,ImageOps
import matplotlib.pyplot as plt
# import itertools
import editdistance
import pickle 
from multiprocessing import Pool

from keras import backend as K
from keras.models import Model
# # from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU,LSTM
from keras.optimizers import SGD,Adadelta
from keras.utils.data_utils import get_file
# # from keras.preprocessing import image
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,TensorBoard
from keras.layers.normalization import BatchNormalization

char_str = '！＂＃＄％＆（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～、。々…‘’“”《》±×÷∧∨∑∏∪∩∈√⊥∥⊙∫∮≡≌≈∽∝≠≤≥∞∵∴℃￡‰§№☆○◎◇□△※→←↑↓①②③④⑤⑥⑦⑧⑨⑩￥℅℉↖↗↘↙▽⊕㎎㎏㎜㎝㎞㎡㏄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐做作坐座丐噩匕夭匾卦偎凫禀谒阱芙茉茴荞荠荸莺蒿薇奕拗捺叽叨吆咧咪唠唧嗦嗤嘁嘀嘹幔岖徙猬馍庵悴愕憔沐涮漩姊嬉缤缭缰玷璧杈桦榄楣榛榕橄橘檐檩昙昵掰牍肴胧腌朦臊飒炫祠盹铐铛锉秕秫鸠鹉鹦瘾癞衩裆蚣蚪蚓蚯蛉蜓蜈蜻蝠蝌蝙蟆螃蟥蟋蟀笙笤筝箫簸翎翩麸跛跷踱蹂躏雳霎鲫鳄鳍鬓啰啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳瞭病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄锗柞珐辊烩硷粳傈镊醛僳酞烃硒矽乂亶亸伋伾俵俶倓倞倴倻偲僇僰剅剕剟勍勩匜卻厾唝啴喆喤嗞嘚嘡噀嚄嚚垍垕垞垟垯垱埇埗埼堉堌堎堲塃塆塮墈墦夬奭姞姽婞婳嫚嫪宬屃峂峃峣峧崟嵎嵖嵚嶓巉帡幪庤庼廋廙弇彧悢惇愔慥憖扅扊扺抃抔拃拤挦捯揳揾搒摽斝旴旸旻昉昽晞晳暅曈杧杻柈栒栟梽梾棁棨棬棻椑楯槚槜橦檵欸殣毐汈汭沄沘沚沨泃泚泜洑洣洨洴洺浉浐浕浡浥涘涢淏湉湜湝湨湲溇溠溦滃滍滘滪滫漹潏潟潵潽澥澴澼瀌瀍灈炟烜烝烺焌焜熜熥燊燏牁牂牚犨狉狝猄猇獏獴玕玙玚玠玡玥玦珣珰珽琇琎琤琫琯瑀瑄瑢璆璈璘璟璠璪瓘瓻甪畯疍疢疭瘆盉盫眊眬矻砮硁硚碶礌礳祃祎祲祼祾禤秾穀穄窅窣窸竑筜筼箓篊篢簃簉簠糵繄纴纻绤绹罽羑翙翚翯耰耲脧腒膙臑臜芼茀茓茝荄荙莙菂菼萩葖蒟蓂蓇蕰藠蘘蜎螠螣蠋袆袗袪袯裈襕觱訄诐豨亍丌兀廿卅丕亘丞鬲孬丨禺丿乇爻卮氐囟胤馗毓睾鼗丶亟鼐乜乩亓芈孛啬嘏仄厍厝厣厥厮靥赝匚叵匦匮赜卣刂刈刎刭刳刿剀剌剞剡剜蒯剽劂劁劐劓冂罔亻仃仉仂仨仡仫仞伛仳伢佤仵伥伧伉伫佞佧攸佚佝佟佗伲伽佶佴侑侉侃侏佾佻侪佼侬侔俦俨俪俅俚俣俜俑俟俸倩偌俳倬倏倮倭俾倜倌倥倨偾偃偕偈偬偻傥傧傩傺僖儆僭僬僦僮儇儋仝氽佘佥俎龠汆籴兮巽黉馘冁夔勹匍訇匐夙兕亠兖亳衮袤亵脔裒嬴蠃羸冫冱冽冼赑赟赪跐凇冖冢冥讠讦讧讪讴讵讷诂诃诋诏诎诒诓诔诖诘诙诜诟诠诤诨诩诮诰诳诶诹诼诿谀谂谄谇谌谏谑谔谕谖谙谛谘谝谟谠谡谥谧谪谫谮谯谲谳谵谶卩卺阝阢阡阪阽阼陂陉陔陟陧陬陲陴隈隍隗隰邗邛邝邙邬邡邴邳邶邺跶踶蹅蹐蹓蹜蹢邸邰郏郅邾郐郄郇郓郦郢郜郗郛郫郯郾鄄鄢鄞鄣鄱鄯鄹酃酆刍奂劢劬劭劾哿勐勖勰叟燮矍廴凵凼鬯厶弁畚巯坌垩垡塾墼壅壑圩圬圪圳圹圮圯坜圻坂坩垅坫垆坼坻坨坭坶坳垭垤垌垲埏垧垴垓垠埕埘埚埙埒垸埴埯埸埤埝蹽蹾堋堍埽埭堀堞堙塄堠塥塬墁墉墚墀馨鼙懿艹艽艿芏芊芨芄芎芑芗芫芸芾芰苈苊苣芘芷芮苋苌苁芩芴芡芪芟苄苎芤苡苷苤茏茇苜苴苒苘茌苻苓茑茚茆茔茕苠苕茜荑荛荜茈莒茼茱莛茯荏荇荃荟荀茗茭茺茳荦荥荨茛荩荬荪荭荮莰莳莴莠莪莓莜莅荼莶莩荽莸荻莘莞莨莼菁萁菥菘堇萘萋菝菽菖萜萸萑萆菔菟萏萃菸菹菪菅菀萦菰菡葜葑葚葙葳蒇蒈葺蒉葸萼葆葩葶蒌蒎萱葭蓁蓍蓐蓦蒽蓓蓊蒺蓠蒡蒹蒴蒗蓥蓣蔌甍蔸蓰蔹蔟蔺轪辀辌辒蕖蔻蓿蓼蕙蕈蕨蕤蕞蕺瞢蕃蕲蕻薤薨薏蕹薮薜薅薹薷薰藓藁藜藿蘧蘅蘩蘖蘼廾弈夼奁耷奚奘匏尢尥尬尴扌扪抟抻拊拚拮挢拶挹捋捃掭揶捱掎掴捭掬掊捩掮掼揲揸揠揿揄揞揎摒揆掾摅摁搋搛搠搌搦搡摞撄摭撖邽摺撷撸撙撺擀擐擗擤擢攉攥攮弋忒甙弑卟叱叩叻吒吖呋呒呓呔呖呃吡呗呙吣吲咂咔呷呱呤咚咛咄呶呦咝哐咭哂咴哒咦哓哔呲咣哕咻咿哌哙哚哜咩咤哝哏哞唛哧哽唔哳唢唣唏唑唪啧喏喵啉啭啁啕唿啐唼郈郚郿鄌鄘唷啖啵啶啷唳唰啜喋嗒喃喱喹喈喁喟啾嗖喑啻嗟喽喾喔喙嗪嗷嗉嘟嗑嗫嗬嗔嗝嗄嗯嗥嗲嗳嗌嗍嗨嗵辔嘞嘈嘌嘤嘣嗾嘧嘭噘噗嘬噍噢噙噜噌噔嚆噤噱噫噻噼嚅嚓嚯囔囗囝囡囵囫囹囿圄圊圉圜帏帙帔帑帱帻帼酦醾帷幄幛幞幡岌屺岍岐岈岘岙岑岚岜岵岢岽岬岫岱岣峁岷峄峒峤峋峥崂崃崧崦崮崤崞崆崛嵘崾崴崽嵬嵛嵯嵝嵫嵋嵊嵩嵴嶂嶙嶝豳嶷巅彳彷徂徇徉後徕徜徨徭徵徼衢彡犭犰犴犷犸狃狁狎狍狒狨狯狩狲狴狷猁狳猃狺狻猗猓猡猊猞猝猕猢猹猥猸猱獐獍獗獠獬獯獾舛夥飧夤夂饣饧饨饩饪饫饬饴饷饽馀馄馇馊馐馑馓馔馕庀庑庋庖庥庠庹庾庳赓廒廑廛廨廪膺忄忉忖忏怃忮怄忡忤忾怅怆忪忭忸怙怵怦怛怏怍怩怫怊怿怡恸恹恻恺恂銍恪恽悖悚悭悝悃悒悌悛惬悻悱惝惘惆惚愠愦愣惴愀愎愫慊慵憬憧憷懔懵忝隳闩闫闱闳闵闶闼闾阃阄阆阈阊阋阌阍阏阒阕阖阗阙阚丬爿戕氵汔汜汊沣沅沔沌汨汩汴汶沆沩泐泔沭泷泸泱泗沲泠泖泺泫泮沱泓泯泾鋆洹洧洌浃浈洇洄洙洎洫浍洮洵洚浏浒浔洳涑浯涞涠浞涓涔浜浠浼浣渚淇淅淞渎涿淠渑淦淝淙渖涫渌渫湮湎湫溲湟溆湓湔渲渥湄滟溱溘滠漭滢溥溧溽溻溷滗溴滏溏滂溟潢潆潇漤漕滹漯漶潋潴漪漉澉澍澌潸潲潼潺濑鍠濉澧澹澶濂濡濮濞濠濯瀚瀣瀛瀹瀵灏灞宀宄宕宓宥宸甯骞搴寤寮褰寰蹇謇辶迓迕迥迮迤迩迦迳迨逅逄逋逦逑逍逖逡逵逶逭逯遄遑遒遐遨遘遢遛暹遴遽邂邈邃邋彐彗彖彘尻咫屐屙孱屣屦羼弪弩弭艴弼鬻屮妁妃妍妩妪妣妗妫妞妤姒妲妯姗妾娅娆姝娈姣姘姹娌娉娲娴娑娣娓婀婧婊婕娼婢婵胬媪媛婷婺媾嫫媲嫒嫔媸嫠嫣嫱嫖嫦嫘嫜嬗嬖嬲嬷孀尕尜孚孥孳孑孓孢驵驷驸驺驿驽骀骁骅骈骊骐骒骓骖骘骛骜骝骟骠骢骣骥骧纟纡纣纥纨纩纭纰纾绀绁绂绉绋绌绐绔绗绛绠绡绨绫绮绯绱绲缍绶绺绻绾缁缂缃缇缈缋缌缏缑缒缗缙缜缛缟缡缢缣缥缦缧缪缫缬缯缱缲缳缵幺畿巛甾邕玎玑玮玢玟珏珂珑玳珀珉珈珥珙顼琊珩珧珞玺珲琏琪瑛琦琥琨琰琮琬钘铻锜琛琚瑁瑜瑗瑕瑙瑷瑭瑾璜璎璀璁璇璋璞璨璩璐瓒璺韪韫韬杌杓杞杩枥枇杪杳枘枧杵枨枞枭枋杷杼柰栉柘栊柩枰栌柙枵柚枳柝栀柃枸柢栎柁柽栲栳桠桡桎桢桄桤梃栝桕桁桧桀栾桊桉栩梵梏桴桷梓桫棂楮棼椟椠棹锧镃镈镋镕镚镠镴镵椤棰椋椁楗棣椐楱椹楠楂楝楫榀榘楸椴槌榇榈槎榉楦楹榧榻榫榭槔榱槁槊槟槠榍槿樯槭樗樘橥槲樾檠橐橛樵檎橹樽樨橼檑檗檫猷獒殁殂殇殄殒殓殍殚殛殡殪轫轭轱轲轳轵轶轸轷轹轺轼轾辁辂辄辇辋闿阇阘辍辎辏辘辚軎戋戗戛戟戢戡戥戤戬臧瓯瓴瓿甏甑甓攴旮旯旰昊杲昃昕昀炅曷昝昴昱昶耆晟晔晁晏晖晡晗晷暄暌暧暝暾曛曜曦曩贲贳贶贻贽赀赅赆赈赉赇赍赕赙觇觊觋觌觎觏觐觑牮犟牝牦牯牾牿犄犋犍犏犒挈挲搿擘耄毪毳毽毵毹氅氇氆氍氕氘氙氚氡氩氤氪氲攵敕敫牒牖爰虢刖肟肜肓肼朊肽肱肫肭肷胨胩胪胛胂胄胙胍胗朐胝胫胱胴胭脍脎胲胼朕脒豚脶脞脬脘脲腈腓腴腙腚腱腠腩腼腽腭腧塍媵膈膂膑滕膣膪臌膻靰靸靺靽靿鞁臁膦欤欷欹歃歆歙飑飓飕飙飚殳彀毂觳斐齑斓於旆旄旃旌旎旒旖炀炜炖炝炻烀炷炱烨烊焐焓焖焯焱煳煜煨煅煲煊煸煺熘熳熵熨熠燠燔燧燹爝爨灬焘煦熹戾戽扃扈扉礻祀祆祉祛祜祓祚祢祗祯祧祺禅禊禚禧禳忑忐鞡鞧鞨韂韨怼恝恚恧恁恙恣悫愆愍慝憩憝懋懑戆肀聿沓泶淼矶矸砀砉砗砘砑斫砭砜砝砹砺砻砟砼砥砬砣砩硎硭硖硗砦硐硇硌硪碛碓碚碇碜碡碣碲碹碥磔磙磉磬磲礅磴礓礤礞礴龛黹黻黼盱眄眍眇眈眚眢眙眭眦眵眸睐睑睇睃睚睨睢睥睿瞍睽瞀瞌瞑瞟瞠瞰瞵瞽町畀畎畋畈畛畲畹疃罘罡罟詈罨罴罱罹羁罾盍盥蠲钅钆钇钋钊钌钍钏钐钔钗钕钚钛钜钣钤钫钪钭钬钯钰钲钴钶钷钸钹钺钼钽钿铄铈铉铊铋铌铍铎铑铒铕铖铗铙铘铞铟铠铢铤铥铧铨铪颋颙飐飔飗铩铫铮铯铳铴铵铷铹铼铽铿锃锂锆锇锊锍锎锏锒锓锔锕锖锘锛锝锞锟锢锪锫锩锬锱锲锴锶锷锸锼锾锿镂锵镄镅镆镉镌镎镏镒镓镔镖镗镘镙镛镞镟镝镡镢镤镥镦镧镨镩镪镫镬镯镱镲镳锺矧矬雉秭秣稆嵇稃稂稞稔饳饸饹饻馃馉稹稷穑黏馥穰皈皎皓皙皤瓞瓠甬鸢鸨鸩鸪鸫鸬鸲鸱鸶鸸鸷鸹鸺鸾鹁鹂鹄鹆鹇鹈鹋鹌鹎鹑鹕鹗鹚鹛鹜鹞鹣鹧鹨鹩鹪鹫鹬鹱鹭鹳疒疔疖疠疝疬疣疳疴疸痄疱疰痃痂痖痍痣痨痦痤痫痧瘃痱痼痿瘐瘀瘅瘌瘗瘊瘥瘘瘕瘙馌瘛瘼瘢瘠癀瘭瘰瘿瘵癃瘳癍癔癜癖癫癯翊竦穸穹窀窆窈窕窦窠窬窨窭窳衤衲衽衿袂袢袷袼裉裢裎裣裥裱褚裼裨裾裰褡褙褓褛褊褴褫褶襁襦襻疋胥皲皴矜耒耔耖耜耠耢耥耦耧耩耨耱耋耵聃聆聍聒聩聱覃顸颀颃颉颌颍颏颔颚颛颞颟颡颢颥颦虍虔虬虮虿虺虼虻蚨蚍蚋蚬蚝蚧蚩蚶蛄蚵蛎蚰蚺蚱蛏蚴蛩蛱蛲蛭蛳蛐蛞蛴蛟蛘蛑蜃蜇蛸蜊蜍蜉蜣蜞蜥蜮蜚蜾蝈蜴蜱蜩蜷蜿螂蜢蝽蝾蝻蝰蝮螋蝓蝣蝼蝤蝥螓螯螨蟒骍骎骙螈螅螭螗螫螬螵螳蟓螽蟑蟊蟛蟪蟠蟮蠖蠓蟾蠊蠛蠡蠹蠼缶罂罄罅舐竺竽笈笃笄笕笊笫笏筇笸笪笮笱笠笥笳笾笞筘筚筅筵筌筠筮筻筢筲筱箐箦箧箸箬箝箨箅箪箜箢箴篑篁篌篝篚篥篦篪簌篾篼簏簖簋鬹魆簟簪簦籁籀臾舁舂舄臬衄舡舢舣舭舯舨舫舸舻舳舴舾艄艉艋艏艚艟艨衾袅袈裘裟襞羝羟羧羯羰羲籼敉粑粝粜粞粢粲粼粽糁糇糌糍糈糅糗糨艮暨羿翕翥翡翦翮翳糸絷綦綮繇纛麴赳趄趔趑趱赧赭豇豉酊酐酎酏酤酢酡酰酩酯酽酾酲酴酹醌醅醐醍醑醢醣醪醭醮醯醵醴醺豕鹾趸跫踅蹙蹩趵趿趼趺跄跖跗跚跞跎跏跆跬跸跣跹跻跤踉跽踔踝踟踬踮踣踯踺蹀踹踵踽蹉蹁蹑蹒蹊蹰蹶蹼蹯蹴躅躔躐躜躞豸貂貊貅貘貔斛觖觞觚觜觥觫觯訾謦靓雩雯霆霁霈霏霪霭霰霾龀龃龅龆龇龈龉龊龌黾鼋鼍隹隼隽雎雒瞿雠銎銮鋈錾鍪鏊鎏鐾鑫鱿鲂鲅鲆鲇鲈稣鲋鲎鲐鲑鲒鲔鲕鲚鲛鲞鲟鲠鲡鲢鲣鲥鲦鲧鲨鲩鲭鲮鲰鲱鲲鲳鲴鲵鲶鲷鲺鲻鲼鲽鳅鳆鳇鳊鳋鱽鱾鲀鲃鲉鲊鲌鲏鲙鲪鲬鲯鲹鲾鳀鳁鳂鳈鳉鳑鳚鳡鳌鳎鳏鳐鳓鳔鳕鳗鳘鳙鳜鳝鳟鳢靼鞅鞑鞒鞔鞯鞫鞣鞲鞴骱骰骷鹘骶骺骼髁髀髅髂髋髌髑魅魃魇魉魈魍魑飨餍餮饕饔髟髡髦髯髫髻髭髹鬈鬏鬟鬣麽麾縻麂麇麈麋麒鏖麝麟黛黜黝黠黟黢黩黧黥黪黯鼢鼬鼯鼹鼷鼽鼾齄鳤鸤鸮鸰鸻鸼鹀鹐鹝鹟鹠鹡鹮鹲黇鼒鼩鼫鼱齁齉龁'

def print_label(label_int):
	#label_int is a list of labels of a text line 
	label_str=[]
	for i in range(0,len(label_int)):
		label_str.append(char_str[label_int[i]-1])
	print(label_str)
def get_label_str(label_int):
	#label_int is a list of labels of a text line 
	label_str=[]
	for i in range(0,len(label_int)):
		label_str.append(char_str[label_int[i]-1])
	return label_str
def plot_char(char,figure_name = 'Original'):#char is the original form with shape 'x,y,...,x,y,65535,0,x,y,...,x,y,65535,0,65535,65535'
	x_list=[]
	y_list=[]
	x_list_array = []
	y_list_array = []
	print('Original length:',int(len(char)/2))
	for i in range(0,int(len(char)/2)):
		if  char[2*i] !=65535 and char[2*i+1]!=65535:
			x_list.append(char[2*i])
			y_list.append(-char[2*i+1])
		else :
			x_list_array.append(x_list)
			y_list_array.append(y_list)
			x_list=[]
			y_list=[]
	plt.figure() 
	for i in range(0,len(x_list_array)):
		plt.plot(x_list_array[i], y_list_array[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()#char 中xy交替出现，65536为结束标识符
def plot_char2(char,figure_name = 'get_sparse_line'):#char[0],char[1] are the x_list and y_list with 65535.
	x=char[0]
	y=char[1]
	x_list=[]
	y_list=[]
	x_list_array = []
	y_list_array = []
	print('sparse length: ',len(char[0]))
	for i in range(0,len(x)):
		if  x[i] !=65535 and y[i]!=65535:
			x_list.append(x[i])
			y_list.append(-y[i])
		else :
			x_list_array.append(x_list)
			y_list_array.append(y_list)
			x_list=[]
			y_list=[]
	print('points num :',len(char[0]))
	plt.figure() 
	for i in range(0,len(x_list_array)):
		plt.plot(x_list_array[i], y_list_array[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()
def plot_char3(char_label,figure_name = 'get_sparse_diff_line'):#char_label = the output of get_sparse_diff_line()
	line = char_label[0]
	print_label(char_label[1])
	x_list = []
	y_list = []

	x = [line[0][0][0]]
	y = [line[0][0][1]]

	chars_point_num = [len(char) for char in line]
	print('get_sparse_diff_line length: ',sum(chars_point_num))
	for char in line:
		for point in char:
			if point[2] ==1:
				x_list.append(x)
				y_list.append(y)
				x=[x[-1]+point[0]]
				y=[y[-1]+point[1]]
			else:
				x.append(x[-1]+point[0])
				y.append(y[-1]+point[1])
	for i in range(len(y_list)):
		for j in range(len(y_list[i])):
			y_list[i][j] = -y_list[i][j]

	plt.figure() 
	for i in range(0,len(x_list)):
		plt.plot(x_list[i], y_list[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()		
def plot_char3_1(char_label,figure_name = 'get_sparse_diff_char'):#char_label = the output of get_sparse_diff_line()
	char = char_label[0]
	print_label(char_label[1])
	x_list = []
	y_list = []

	x = [char[0][0]]
	y = [char[0][1]]

	# chars_point_num = [len(char) for char in line]
	print('get_sparse_diff_line length: ',len(char))

	for point in char:
		if point[2] !=0:
			x_list.append(x)
			y_list.append(y)
			x=[x[-1]+point[0]]
			y=[y[-1]+point[1]]
		else:
			x.append(x[-1]+point[0])
			y.append(y[-1]+point[1])

	for i in range(len(y_list)):
		for j in range(len(y_list[i])):
			y_list[i][j] = -y_list[i][j]

	plt.figure() 
	for i in range(0,len(x_list)):
		plt.plot(x_list[i], y_list[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()		
def plot_char4(diff_data,figure_name = 'diff'):#diff_data is a textline data after diff(), with shape [[delta_x,delta_y,state]*points_num]
	x_list = []
	y_list = []
	x = [diff_data[0][0]]
	y = [diff_data[0][1]]
	print('diff length: ',len(diff_data))
	for point in diff_data:
		if point[2] == 1:
			x_list.append(x)
			y_list.append(y)
			x=[x[-1]+point[0]]
			y=[y[-1]+point[1]]
		else:
			x.append(x[-1]+point[0])
			y.append(y[-1]+point[1])
	for i in range(len(y_list)):
		for j in range(len(y_list[i])):
			y_list[i][j] = -y_list[i][j]

	plt.figure() 
	for i in range(0,len(x_list)):
		plt.plot(x_list[i], y_list[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()	
def plot_char5(data_label,figure_name = 'generated sample'):
	# data_label is a textline after sample_generator(), with shape [[[delta_x,delta_y,state]*points_num],label_int listof the line]
	x_list = []
	y_list = []
	diff_data = data_label[0]
	print_label( data_label[1])
	x = [diff_data[0][0]]
	y = [diff_data[0][1]]
	print('generated sample length: ',len(diff_data))
	for point in diff_data:
		if point[2] == 1:
			x_list.append(x)
			y_list.append(y)
			x=[x[-1]+point[0]]
			y=[y[-1]+point[1]]
		else:
			x.append(x[-1]+point[0])
			y.append(y[-1]+point[1])
	for i in range(len(y_list)):
		for j in range(len(y_list[i])):
			y_list[i][j] = -y_list[i][j]

	plt.figure() 
	for i in range(0,len(x_list)):
		plt.plot(x_list[i], y_list[i], color='b', linewidth=1, alpha=0.6)
		# plt.scatter(x_list_array[i], y_list_array[i], color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.axis('equal')
	plt.title(figure_name)
	plt.show()	


def get_sparse_line(char, TdistConst= 8,TcosConst = 0.99):#return sparse_line_x,sparse_line_y,chars_point_num,	chars_left_bound, chars_right_bound
	#char is the original form with shape 'x,y,...,x,y,65535,0,x,y,...,x,y,65535,0,65535,65535'
	sparse_line_x =[]#textline的稀疏化后的x坐标
	sparse_line_y = []#textline的稀疏化后的y坐标
	chars_point_num = []#稀疏化后的textline中每个字的采样点数
	chars_left_bound =[]#稀疏化后的textline中每个字的最左端值
	chars_right_bound = []#稀疏化后的每个字的最右端值
	strokes_ritght_bound = []#临时变量
	stroke_point_num = 0
	char_point_num = 0

	for i in range(0,int(len(char)/2)):
		if char[2*i]==65535 :
			sparse_line_x.append(char[2*i])
			sparse_line_y.append(char[2*i+1])
			char_point_num+=1
			stroke_point_num+=1
			if stroke_point_num>1:
				strokes_ritght_bound.append(max(sparse_line_x[-stroke_point_num:-1]))
			stroke_point_num = 0
			if char[2*i+1]==65535:
				chars_point_num.append(char_point_num)	
				if char_point_num<2:
					print ('line 101:',char[2*i-4:2*i+4])
				chars_left_bound.append(min(sparse_line_x[-char_point_num:-1]))
				chars_right_bound.append(max(strokes_ritght_bound[:]))
				strokes_ritght_bound = []
				char_point_num = 0
		elif stroke_point_num==0:
			sparse_line_x.append(char[2*i])
			sparse_line_y.append(char[2*i+1])
			char_point_num+=1
			stroke_point_num+=1
		else:
			delta_xi=char[2*i+2]-char[2*i]
			delta_xim = char[2*i]-sparse_line_x[-1]
			delta_yi = char[2*i+3]-char[2*i+1]
			delta_yim = char[2*i+1]-sparse_line_y[-1]
			
			T_dist_m = math.sqrt(pow(delta_xim,2)+pow(delta_yim,2))
			T_dist = math.sqrt(pow(delta_xi,2)+pow(delta_yi,2))
			T_cos = (delta_xi*delta_xim+delta_yi*delta_yim)/(T_dist*T_dist_m+0.000001)
			if(T_dist_m>TdistConst and T_cos<TcosConst):
				sparse_line_x.append(char[2*i])
				sparse_line_y.append(char[2*i+1])
				char_point_num+=1
				stroke_point_num+=1
	# print(len(sparse_line_x),len(sparse_line_y),len(chars_point_num), len(chars_left_bound) ,len (chars_right_bound))
	assert(len(sparse_line_x)==len(sparse_line_y))
	assert(len(chars_point_num)== len(chars_left_bound) and len(chars_left_bound)== len (chars_right_bound) and  len(chars_left_bound))
	# print('sparse line points num : ',len(sparse_line_x))
	return sparse_line_x,sparse_line_y,chars_point_num,	chars_left_bound, chars_right_bound
def diff(x_list,y_list,time_dense_size=160): #return [[diff_x, diff_y, pen_state]*len(x_list)]
	channel=3
	diff_data = np.zeros([len(x_list),channel])#up:1,down:0
	skip_count = 0
	x_min = x_list[0]
	x_max = x_list[0]
	for i in range(1,len(x_list)):
		if x_list[i]!=65535:
			# print(i,i-1-skip_count)
			diff_data[i][0] = x_list[i]-x_list[i-1-skip_count]
			diff_data[i][1] = y_list[i]-y_list[i-1-skip_count]
			diff_data[i][2] = int(bool(skip_count))
			skip_count = 0
			x_min = min(x_min,x_list[i])
			x_max = max(x_max,x_list[i])
		else:
			skip_count += 1
	diff_data[len(x_list)-1]=[x_max-x_list[-3]+x_list[0]-x_min,y_list[-3]-y_list[0],1]
	return diff_data
def move_x(x_list,x_move): 
	#used for sample_generater if move the chars on the begin to the tail of the line
	#x_move:means the 'x's of  each sample-x_move，might be eighter x_min of each char or (-(chars_right_bound[chars_num-1]-first_x_min)) of the text line.
	for i in range(0,len(x_list)):
		if x_list[i]!=65535:
			x_list[i] =x_list[i]-x_move
	return x_list
def sample_generator0(char,label,absolute_max_label_len,time_dense_size=160):# return sample_x_list,sample_y_list,sample_label_list
	#char need a structure of '[0]sparse_line_x,[1]sparse_line_y,[2]chars_point_num,[3]chars_left_bound, [4]chars_right_bound'
	# def sample_generator(x,y,chars_point_num,chars_left_bound,chars_right_bound,label,time_dense_size=300):
	x=char[0]
	y=char[1]
	chars_point_num=char[2]
	chars_left_bound=char[3]
	chars_right_bound=char[4]
	if(len(chars_point_num) != len(chars_left_bound) or len(chars_point_num) == 0):
		print('len(chars_point_num):',len(chars_point_num) )
		print(len(chars_left_bound) )
	chars_num = len(chars_point_num)

	char_index = 0
	if sum(chars_point_num)>time_dense_size and chars_num>1:
		char_index = random.randint(0,chars_num-1)
	
	sample_x_list = []
	sample_y_list = []
	sample_label_list = []

	first_x_min = chars_left_bound[char_index]
	x_move = first_x_min #意味着每个sample的x都减去这个数，可能是每个字的x_min,#也可能是整个文本行的-(chars_right_bound[chars_num-1]-first_x_min)

	while (len(sample_x_list)+chars_point_num[char_index]) <= time_dense_size and len(sample_label)<absolute_max_label_len:
		# points_catch += chars_point_num[char_index]
		if char_index==chars_num-1:
			left_index = -chars_point_num[char_index]
			right_index = None
		else:
			left_index = sum(chars_point_num[:(char_index)])
			right_index = left_index+chars_point_num[char_index]-1

		if right_index-left_index>3:
			sample_x_list += move_x(x[left_index:right_index],x_move)
			sample_y_list += y[left_index:right_index]
			sample_label_list.append(label[char_index])

		if char_index<chars_num-1:
			char_index += 1
		else:
			char_index = 0
			x_move = -(chars_right_bound[chars_num-1]-first_x_min)
	return sample_x_list,sample_y_list,sample_label_list

def get_sparse_diff_line(data_label, TdistConst= 8,TcosConst = 0.99):#return char_data(x,y,state),label
	#char is the original form with shape 'x,y,...,x,y,65535,0,x,y,...,x,y,65535,0,65535,65535'
	# def get_sparse_diff_line(data_label,data_sparse_diff_label ,TdistConst= 8,TcosConst = 0.99):#return char_data(x,y,state),chars_point_num
	line = data_label[0]
	assert(int(len(line)/2)>3)
	line_sparse_diff = []#textline的稀疏化后的[[x，y，state ]*(sum(char_point_num)+2)],右端state=-1，记录了第一个字符移动到后面的应该有的差分值
	previous_x = line[0]
	previous_y = line[1]	
	char_sparse_diff =[]
	stroke_point_count = 0
	skip_count = 0
	x_first = line[0]
	x_min = x_first
	x_last = line[len(line)-6]
	x_max = x_last
	for i in range(2,int(len(line)/2)-1):
		if line[2*i]==65535 :
			if line[2*i+1]!=65535:
				if stroke_point_count>=2:
					char_sparse_diff.append([line[2*i-2]-previous_x,line[2*i-1]-previous_y,int(bool(skip_count))])
					previous_x = line[2*i-2]
					previous_y = line[2*i-1]
			skip_count += 2
			stroke_point_count = 0
		else:
			x_min=min(line[2*i],x_min)
			x_max=max(line[2*i],x_max)
			if skip_count>0 or line[2*i-2]==65535:
				char_sparse_diff.append([line[2*i]-previous_x,line[2*i+1]-previous_y,1])
				previous_x = line[2*i]
				previous_y = line[2*i+1]
				if line[2*i-1]==65535:
					line_sparse_diff.append(char_sparse_diff)
					char_sparse_diff=[]
				skip_count=0
			else:
				delta_xi=line[2*i]-line[2*i-2]
				delta_xim = line[2*i-2]-previous_x
				delta_yi = line[2*i+1]-line[2*i-1]
				delta_yim = line[2*i-1]-previous_y
			
				T_dist_m = math.sqrt(pow(delta_xim,2)+pow(delta_yim,2))
				T_dist = math.sqrt(pow(delta_xi,2)+pow(delta_yi,2))
				T_cos = (delta_xi*delta_xim+delta_yi*delta_yim)/(T_dist*T_dist_m+0.0001)
				if(T_dist_m>TdistConst and T_cos<TcosConst):
					char_sparse_diff.append([delta_xim,delta_yim,int(bool(skip_count))])
					previous_x = line[2*i-2]
					previous_y = line[2*i-1]
			stroke_point_count+=1
			skip_count=0
	char_sparse_diff.append([x_max-x_last+x_first-x_min,line[1]-line[len(line)-5],1])
	line_sparse_diff.append(char_sparse_diff)
	return [line_sparse_diff,data_label[1]]
def get_sparse_diff_char(data_label, TdistConst= 8,TcosConst = 0.99):
	char = data_label[0]
	x_first = char[0]
	y_first = char[1]
	x_min = x_first
	x_last = char[len(char)-6]
	y_last = char[len(char)-5]
	x_max = x_last
	previous_x = char[0]
	previous_y = char[1]
	char_sparse_diff =[]
	stroke_point_count = 0
	skip_count = 0
	for i in range(int(len(char)/2)):
		if char[2*i]==65535 :
			if char[2*i+1]!=65535:
				if stroke_point_count>=2:
					char_sparse_diff.append([char[2*i-2]-previous_x,char[2*i-1]-previous_y,int(bool(skip_count))])
					previous_x = char[2*i-2]
					previous_y = char[2*i-1]
			skip_count += 2
			stroke_point_count = 0
		else:
			x_min=min(char[2*i],x_min)
			x_max=max(char[2*i],x_max)
			if skip_count>0 or char[2*i-2]==65535:
				char_sparse_diff.append([char[2*i]-previous_x,char[2*i+1]-previous_y,1])
				previous_x = char[2*i]
				previous_y = char[2*i+1]
				# if char[2*i-1]==65535:
				# 	char_sparse_diff.append(char_sparse_diff)
				# 	char_sparse_diff=[]
				skip_count=0
			else:
				# print(char[2*i],char[2*i-2])
				delta_xi = char[2*i]-char[2*i-2]
				delta_xim = char[2*i-2]-previous_x
				delta_yi = char[2*i+1]-char[2*i-1]
				delta_yim = char[2*i-1]-previous_y
			
				T_dist_m = math.sqrt(pow(delta_xim,2)+pow(delta_yim,2))
				T_dist = math.sqrt(pow(delta_xi,2)+pow(delta_yi,2))
				T_cos = (delta_xi*delta_xim+delta_yi*delta_yim)/(T_dist*T_dist_m+0.0001)
				if(T_dist_m>TdistConst and T_cos<TcosConst):
					char_sparse_diff.append([delta_xim,delta_yim,int(bool(skip_count))])
					previous_x = char[2*i-2]
					previous_y = char[2*i-1]
			stroke_point_count+=1
			skip_count=0
	char_sparse_diff.append([x_max-x_last,y_last,-1])
	char_sparse_diff.append([x_first-x_min,y_first,-1])
	#x2_first-(x2_min-x1_max)-x1_last = (x2_first-x2_min)+(x1_max-x1_last)
	# print('442:char_sparse_diff',char_sparse_diff)
	return [char_sparse_diff,data_label[1]]

def sample_generator(data_label,absolute_max_label_len,time_dense_size,channel = 3):
	#data_label=[[x,y,state],label],  return [[x,y,state],label]
	data = data_label[0]
	label = data_label[1]
	chars_point_num=[len(data[i]) for i in range(0,len(data))]
	chars_num = len(chars_point_num)

	char_index = 0
	if sum(chars_point_num)>time_dense_size and chars_num>1:
		char_index = random.randint(0,chars_num-1)

	sample_data = np.zeros([time_dense_size, channel])
	sample_label = []
	point_count=0

	while point_count+chars_point_num[char_index] <= time_dense_size and len(sample_label)<absolute_max_label_len :
		sample_data[point_count:point_count+chars_point_num[char_index]] = data[char_index][:]
		sample_label.append(label[char_index])
		point_count+=chars_point_num[char_index]

		if char_index<chars_num-1:
			char_index += 1
		else:
			char_index = 0
	sample_data[-1] = [0,0,1]
	# print('470:',sample_data[-10:])
	return sample_data,sample_label
def sample_generator_group(dataset,absolute_max_label_len,time_dense_size,channel = 3):
	sample_data = np.zeros( [time_dense_size, channel])
	sample_label = []
	point_count=0
	chars_num = len(dataset)
	random_index = random.randint(0,chars_num-1)
	note = [0,0,0]

	while point_count+len(dataset[random_index][0])<time_dense_size and len(sample_label)<absolute_max_label_len:

		# plot_char3_1(dataset[random_index])

		sample_data[point_count:point_count+len(dataset[random_index][0])-1] = dataset[random_index][0][:-1]
		note = sample_data[point_count-1]
		if point_count>0:
			sample_data[point_count-1] = [note[0]+dataset[random_index][0][-1][0],dataset[random_index][0][-1][1]-note[1],1]
		sample_label.append(dataset[random_index][1][0])
		point_count = point_count+len(dataset[random_index][0])-1
		random_index = random.randint(0,chars_num-1)
	sample_data[-1] = [0,0,1]
	return sample_data,sample_label


def read_pickle_dataset(file_dir):
	with open(file_dir,'rb') as file :
		data_label_set = pickle.load(file)
	return data_label_set
def processing_write_char_set(data_dir,label_dir,file_name,time_dense_size, start = None, end = None):
	print('reading & sparsing data',data_dir,'from',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'),'......')
	data_int = np.fromfile(data_dir,'int32')
	data_label_list = [] 
	point_num_sum =0
	with open(label_dir,'rt') as f:
		lines = f.readlines()
		for line in lines:
			elements = [int(x) for x in (line.split(','))]
			if elements[0]>3:
				data_label_list.append([data_int[(point_num_sum*2):((point_num_sum+elements[0])*2)],[elements[1]+1]]) 
			else:
				print(label_dir,elements,data_int[(point_num_sum*2):((point_num_sum+elements[0])*2)])
			point_num_sum += elements[0]
	print('finish reading at',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

	# print( data_label_list[3000])
	# plot_char(data_label_list[3000][0])
	# sparse_diff = get_sparse_diff_char(data_label_list[3000])
	# plot_char3_1(sparse_diff)
	# print(sparse_diff)
	print('sparsing & diff data from',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
	agents =10
	chunksize = 9
	with Pool(processes = agents) as pool:
		data_sparse_diff_label = pool.map(get_sparse_diff_char, data_label_list[start:end], chunksize )
	print('finish sparsing & diff data at ',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'),', begin write file:')
	# a_sample = sample_generator_group(dataset=data_sparse_diff_label,absolute_max_label_len=20,time_dense_size=time_dense_size)
	# plot_char5(a_sample)
	# input('wait')\
	print(data_sparse_diff_label)
	print('line529:',len(data_sparse_diff_label))

	with open(file_name,'wb') as file:
		pickle.dump(data_sparse_diff_label, file,1)
	print('finish write to file at ',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
	#data_sparse_diff_label：[sample_num*[[sparse_diff_points_num*[delta_x,delta_y,state]],[chars_labels]]]
	return #data_list,label_int_list#,max_label_len,label_str_list
def processing_write_textline_set(data_dir,label_dir,file_name,start = None,end = None):#,data_size,time_dense_size) return data_list(每一个元素list里面包含x,y,65535),label_int_list
	print('reading & sparsing data',data_dir,'from',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'),'......')
	data_int = np.fromfile(data_dir,'int32')
	point_num_sum = 0
	data_label_list = []

	tmp = 0
	with open(label_dir,'rt') as f:
		lines = f.readlines()
		for line in lines:
			elements = [int(x) for x in (line.split(','))]
			if elements[0]>3:
				data_label_list.append([data_int[(point_num_sum*2):((point_num_sum+elements[0])*2)],elements[1:]]) 
			else:
				print(label_dir,elements,data_int[(point_num_sum*2):((point_num_sum+elements[0])*2)])
			point_num_sum += elements[0]

	print('finish reading at',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
	print('sparsing & diff data from',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
	# for i in range(32*560,32*569,):
	# 	# char = data_label_list[i][0]
	# 	# plot_char(data_label_list[i][0])
	# 	# print(data_label_list[i][1])
	# 	# sparse = get_sparse_line(char)
	# # 	plot_char2 (sparse)
	# 	# diff_char = diff(sparse[0],sparse[1])
	# # 	# print(sparse[0][:3])
	# # 	print_label(data_label_list[i][1])
	# # 	plot_char4(diff_char)

	# sparse_diff = get_sparse_diff_line(data_label_list[3])
	# print(sparse_diff)
	# 	# plot_char3(sparse_diff)
	# 	# print_label(sparse_diff[1])
	# 	# print('data_label_list[i][1]', data_label_list[i][1])
	# 	sample_data = sample_generator(sparse_diff)
	# 	# plot_char5(sample_data)
	# input('wait')
	agents =10
	chunksize = 9

	with Pool(processes = agents) as pool:
		data_sparse_diff_label = pool.map(get_sparse_diff_line, data_label_list[start:end], chunksize )
	print('finish sparsing & diff data at ',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

	with open(file_name,'wb') as file:
		pickle.dump(data_sparse_diff_label, file,1)
	print('finish write to file at ',datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))
	#data_sparse_diff_label has a strugle of:[line_num*[chars_num_in_line*[[sparse_diff_points_num*[delta_x,delta_y,state]]],[chars_labels]]]
	return #data_list,label_int_list#,max_label_len,label_str_list

def ctc_lambda_func(args):
	y_pred, labels,input_length, label_length = args		
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def decode_batch(test_func, word_batch):
	out = test_func([word_batch, 0])[0]
	ret = []
	for j in range(out.shape[0]):
		out_best = [k for k, g in itertools.groupby(list(np.argmax(out[j, 2:], 1)))]
		outstr=[char_str[c-1] for c in out_best]
		ret.append(outstr)
		if ret[0]==' ':
			ret.pop(0)
		if ret[len(ret)-1]==' ':
			ret.pop
	return ret
class batch_generator(Callback): #dataset:[0]xy data with 65535,[1]label_int
	def __init__(self, data_label_list, batch_size = 32, label_classes = len(char_str)+1,absolute_max_label_len = 20,
				 channel =3 ,time_dense_size = 160,char_data_label=None):

		textline_num = len(data_label_list)
		if  textline_num% batch_size!=0:
			repeat_num = batch_size-textline_num % batch_size
			random_textline_index = [random.randint(0,textline_num-1) for i in range(repeat_num)]
			print('random repeted img: ',random_textline_index)
			
			for i in range(0,len(random_textline_index)):
				data_label_list.append(data_label_list[random_textline_index[i]])

		self.dataset_size = len(data_label_list)
		self.data_label_list = data_label_list

		assert self.dataset_size%batch_size ==0
		self.char_data_label = char_data_label

		self.absolute_max_label_len = absolute_max_label_len
		self.channel = channel
		self.batch_size = batch_size
		self.time_dense_size = time_dense_size
		self.label_classes = label_classes
		self.cur_sample_index = 0
		self.blank_label = 0
	
	
	def get_output_size(self):
		#for outClass function use
		return self.label_classes
	def get_dataset_size(self):
		return self.dataset_size
	def get_batch(self,  cur_sample_index, train=True):
		assert(K.image_data_format() != 'channels_first')
		X_data = np.zeros([self.batch_size, self.time_dense_size, self.channel])
		mix_img = [[' ']*self.channel]*self.time_dense_size

		input_length = np.zeros([self.batch_size, 1])
		labels = np.zeros([self.batch_size, self.absolute_max_label_len])
		label_length = np.zeros([self.batch_size, 1])
		source_str = []
		sample_count = 0

		for i in range(0, self.batch_size):
			# Mix in some blank inputs.  This seems to be important :破坏一些label，据说对训练集好
			# achieving translational invariance
			#Y_len：#每个true label的长度
			if train and i > self.batch_size:
				X_data[i:,:] = mix_img[:]
				labels[i, 0] = self.blank_label
				input_length[i] = self.time_dense_size
				label_length[i] = 1	 
				source_str.append('') 				
			else:
				if not train:
					select_line_sample = 1
				else:
					select_line_sample = random.randint(0,1)
				if select_line_sample:
					sample_data,sample_label = sample_generator(data_label=self.data_label_list[cur_sample_index+i],
														absolute_max_label_len=self.absolute_max_label_len, time_dense_size=self.time_dense_size,
														channel = self.channel)
					sample_count+=1
				else:
					sample_data,sample_label = sample_generator_group(dataset=self.char_data_label,absolute_max_label_len=self.absolute_max_label_len,
																		time_dense_size=self.time_dense_size,channel = self.channel)
				if len(sample_data)>0 and len(sample_label)>0:
					X_data[i,:len(sample_data)] = sample_data[:]
					labels[i,:len(sample_label)]= sample_label[:]
				else:
					X_data[i,:]=X_data[i-1,:]
					labels[i,:]=X_data[i-1,:]

				input_length[i] = self.time_dense_size-2 #ctc的time_dim
				label_length[i] = len(sample_label)
				source_str.append(get_label_str(sample_label))

		inputs= {'the_input': X_data,
					'the_labels': labels,
					'input_length': input_length,
					'label_length': label_length,
					'source_str': source_str  # used for visualization only
				}
		outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
		return (inputs, outputs),sample_count
	def next_train(self):
		while 1:
			if self.cur_sample_index+self.batch_size >= self.dataset_size:
				self.cur_sample_index = (self.cur_sample_index+self.batch_size) % self.dataset_size
			ret,sample_count = self.get_batch(self.cur_sample_index, train=True)
			self.cur_sample_index += sample_count
			yield ret
	def next_val(self):
		while 1:
			ret = self.get_batch(self.cur_sample_index, train=False)
			self.cur_sample_index += self.batch_size
			if self.cur_sample_index >= self.dataset_size:
				self.cur_sample_index = self.dataset_size % self.cur_sample_index
			yield ret

class VizCallback(Callback):
	def __init__(self, out_txt_path, model_file_dir,test_func,train_sample_gen,train_set_size,
				validation_sample_gen,validation_set_size,initial_epoch,num_display_words = 6):
		self.out_txt_path = out_txt_path
		self.model_file_dir = model_file_dir
		self.train_sample_gen =train_sample_gen
		self.train_set_size = train_set_size
		self.validation_sample_gen = validation_sample_gen
		self.validation_set_size = validation_set_size
		self.num_display_words = num_display_words
		self.test_func =test_func
		self.start_epoch = initial_epoch

	def show_edit_distance(self, num, data_gen): #用友元函数实现比较好
		num_left = num
		mean_norm_ed = 0.0
		mean_ed = 0.0
		while num_left > 0:
			word_batch = next(data_gen)[0]
			num_proc = min(word_batch['the_input'].shape[0], num_left)
			decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
			for j in range(0, num_proc):
				# if decoded_res[j]!=word_batch['source_str'][j]:
				# 	print('pred  :',decoded_res[j])
				# 	print('label :',word_batch['source_str'][j])
				# input()
				edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
				mean_ed += float(edit_dist)
				mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
			num_left -= num_proc

		mean_norm_ed = mean_norm_ed / num
		mean_ed = mean_ed / num
		print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
				% (num, mean_ed, mean_norm_ed))
		return mean_norm_ed
	def on_train_begin(self, logs={}):  
		self.out_file = open(self.out_txt_path,'a')
		# self.train_loss_file = open('./Train_loss.txt')
		# self.test_loss_file = open('./Test_loss.txt')
		# self.train_acc_file = open('./Train_acc.txt')
		self.test_acc_file = open('./Test_acc.txt','a')
		run_time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')#调用系统时间计时
		# print('run time from: ',run_time)
		self.out_file.write(run_time+'\t'+'start_epoch = %02d' % (self.start_epoch)+'\n')
		self.out_file.write('epoch\t\tt_loss\t\tt_acc\t\tv_loss\t\tv_acc'+'\n')  
		# self.train_loss_file.write('train_loss'+'\n')
		# self.test_loss_file.write('test_loss'+'\n')
		self.test_acc_file.write(run_time+'\t'+'start_epoch = %02d' % (self.start_epoch))
		self.test_acc_file.write('\n'+'test_acc'+'\n')
		if self.start_epoch != 0:
			val_acc = str(1-self.show_edit_distance(self.validation_set_size,self.validation_sample_gen))[:6]
			self.test_acc_file.write(val_acc+'\t')

	def on_epoch_end(self, epoch, logs={}):
		loss = str(logs.get('loss'))[:6]
		acc = str(1-self.show_edit_distance(self.train_set_size,self.train_sample_gen))[:6]
		val_loss = str(logs.get('val_loss'))[:6]	
		val_acc = str(1-self.show_edit_distance(self.validation_set_size,self.validation_sample_gen))[:6]

		self.out_file.write(str(epoch)+'\t\t'+loss+'\t\t'+acc +'\t\t'+val_loss+'\t\t'+val_acc+'\n')
		self.test_acc_file.write(val_acc+'\t')
		self.model.save_weights(os.path.join(self.model_file_dir, 'credit_weights%02d.h5' % (epoch)))
	def on_train_end(self, logs={}):
		self.out_file.close()
		self.test_acc_file.close()

def train(start_epoch, stop_epoch):
	minibatch_size = 32
	absolute_max_label_length = 15
	# max_epoch = stop_epoch
	label_class_num =len(char_str)+1# 7357 #len(label_index_list.keys())+1

	model_file_dir = './'
	record_txt_name = './accuracy_loss.txt'

	channel=3
	time_dense_size = 160 #49 # the length of the input to lstm network, eqs to the width of the feature map(the height is 1)
	rnn_size = (32,256)
	dropout = (0.1,0.2,0.5)

	train_dir = '../data/train.bin'
	test_dir = '../data/com.bin'
	train_label_dir = '../data/train.txt'
	test_label_dir = '../data/com.txt'
	char_dir = '../data/char_point.bin'
	char_label_dir = '../data/char_label.txt'
	# train_sparse_diff_data_dir = '../data/train_sparse_diff.txt'
	char_sparse_diff_data_dir = '../data/char_sparse_diff.txt'
	test_sparse_diff_data_dir = '../data/test_sparse_diff.txt'
	train_sparse_diff_data_dir = 'train_sparse_diff.txt'
	# char_sparse_diff_data_dir = 'char_sparse_diff.txt'
	# processing_write_textline_set(train_dir,train_label_dir,train_sparse_diff_data_dir,32*560,32*600)#,train_set_size,time_dense_size)
	# processing_write_textline_set(test_dir,test_label_dir,test_sparse_diff_data_dir)#,tesize,time_dense_size)
	processing_write_char_set(char_dir,char_label_dir,char_sparse_diff_data_dir,time_dense_size = time_dense_size)
	# processing_write_char_set(char_dir,char_label_dir,char_sparse_diff_data_dir,time_dense_size = time_dense_size,start =0,end = 1000)#,tesize,time_dense_size)
	input('wait 798')
	print('reading from:', datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))

	data_trainset = read_pickle_dataset(train_sparse_diff_data_dir)
	data_charset = read_pickle_dataset(char_sparse_diff_data_dir)
	data_testset = read_pickle_dataset(test_sparse_diff_data_dir)
	print('end at:', datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'))


	train_data = batch_generator(data_label_list=data_trainset, batch_size = minibatch_size, label_classes = label_class_num, 
					absolute_max_label_len = absolute_max_label_length, channel =channel, time_dense_size = time_dense_size,
					char_data_label = data_charset)
	test_data = batch_generator(data_label_list=data_testset, batch_size = minibatch_size, label_classes = label_class_num, 
					absolute_max_label_len = absolute_max_label_length, channel =channel, time_dense_size = time_dense_size)
	assert(K.image_data_format() != 'channels_first')

	act = 'relu'
	input_shape = (time_dense_size,channel)

	input_data = Input(name='the_input',shape = input_shape, dtype='float32')
	lstm = LSTM(rnn_size[0], return_sequences=True, kernel_initializer='he_normal')(input_data)
	lstm = LSTM(rnn_size[1], return_sequences=True, kernel_initializer='he_normal')(lstm)
	inner = Dense(train_data.get_output_size(), kernel_initializer='he_normal',name='dense1',trainable=True)(lstm)
	y_pred = Activation('softmax', name='softmax')(inner)

	Model(inputs=input_data, outputs=y_pred).summary()
	labels = Input(name='the_labels', shape=[train_data.absolute_max_label_len], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,labels, input_length, label_length])

	# clipnorm seems to speeds up convergence lr=0.02
	# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	adadelta = Adadelta(lr=1.0,rho=0.95,epsilon =1e-06)

	#定义整个模型：
	model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

	#编译模型
	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adadelta)
	if start_epoch > 0:
		weight_file =  os.path.join(model_file_dir, 'credit_weights%02d.h5' % (start_epoch))
		model.load_weights(weight_file)

	# captures output of softmax so we can decode the output
	#用于可视化结果 
	# model_file_path = './model_{epoch:02d}.h5'
	# out_txt_path = './model_{epoch:02d}.txt'
	test_func = K.function([input_data, K.learning_phase()], [y_pred])

	record = VizCallback(out_txt_path = record_txt_name , model_file_dir = model_file_dir,test_func=test_func,
						train_sample_gen = train_data.next_train(),train_set_size=train_data.get_dataset_size(),
						validation_sample_gen = test_data.next_val(),validation_set_size=test_data.get_dataset_size(),
						initial_epoch=start_epoch)
	model.fit_generator(generator=train_data.next_train(), steps_per_epoch=1391*2,#1391,int(train_data.get_dataset_size()/minibatch_size),
						epochs=stop_epoch, validation_data=test_data.next_val(), 
						validation_steps=int(test_data.get_dataset_size()/minibatch_size+1),#shuffle=True,
						callbacks=[record,TensorBoard(log_dir='./gragh/', histogram_freq=0)], initial_epoch=start_epoch)	


	print(start_epoch,'-',stop_epoch,',exit')


if __name__ == '__main__':
	run_time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')#调用系统时间计时
	print('run time from: ',run_time)

	train(0, 10000)	
