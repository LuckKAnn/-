import string
import pymysql
import random

"""
    数据库操纵工具类: 负责数据库表的创建，以及相应行李信息的插入，查询
"""
class DbUtils:
    def __init__(self):
        # 数据库连接
        self.db = pymysql.connect(host = "101.200.87.37",user = "root",password = "123456",database = "suitcase_detection_schema")
        # 创建游标对象 cursor
        cursor = self.db.cursor()
        # 如果表存在则删除
        cursor.execute("DROP TABLE IF EXISTS suitcaseInfo")
        # 创建表
        sql = """CREATE TABLE suitcaseInfo (
                suitcase_no varchar(255) NOT NULL,
                primary key (suitcase_no),
                 flight_id varchar(255) NOT NULL,
                 flight_dst varchar(255) NOT NULL ,
                 exit_port CHAR NOT NULL )"""
        cursor.execute(sql)
        self.db.commit()
        cursor.close()


    """
        随机生成行李航班信息
        包括: 行李的航班号，目的地，分拣口
    """
    def generateRandomInfo(self):
        flight_id = str(random.choice("RSTUVWXYZ")) + str(random.randint(1000, 9999))
        flight_dst = random.choice(
            ['贵州', "成都", '郑州', "洛阳", '焦作', "商丘", '南阳', "福州", '合肥', "厦门", '泉州', "兰州", '遵义', "安顺", '海口', "三亚",
             '石家庄', "哈尔滨", '保定', "唐山", '武汉', "荆门", '长沙', "长春", '南京', "南昌", '沈阳', "济南", '西宁', "太原", '西安', "杭州", ])
        exit = str(random.choice(string.ascii_uppercase))
        return  flight_id,flight_dst,exit

    """
        插入行李航班信息
        参数:
            sno: 行李编号. 一般由秒级时间戳+航班id+分拣口编号组成
            flight_id 航班id
            flight_dst 航班目的地
            exit 分拣口
        返回值: None
    """
    def insertSuitcaseInfo(self,sno,flight_id,flight_dst,exit):
        try:
            cursor = self.db.cursor()
            # 使用预处理语句创建表
            sql = """INSERT INTO suitcaseInfo(suitcase_no,
             flight_id, flight_dst, exit_port)
             VALUES ('""" + sno+"','"+ flight_id +"','"+ flight_dst+"','"+exit+"')"
            # print("插入数据"+sno+" "+flight_id+" "+flight_dst+" "+exit)
            cursor.execute(sql)
            self.db.commit()
            cursor.close()
        except Exception as e:
            print(e)

    """
         通过行李编号，查询相应的航班信息。用于在行李匹配之后，查询匹配行李的信息，赋给当前行李
         参数:
             sno: 行李编号. 一般由秒级时间戳+航班id+分拣口编号组成
        返回值:
            行李编号，航班id，目的地，分拣口
     """
    def querySuitcaseInfo(self, sno):
        try:
            cursor = self.db.cursor()
            # 使用预处理语句创建表
            sql = """SELECT * FROM suitcaseInfo
            WHERE  suitcase_no = '%s' """ %(sno)
            # print(sql)
            cursor.execute(sql)
            result = cursor.fetchone()
            # print(result)
            cursor.close()
            return  result[0],result[1],result[2],result[3]

        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        db = DbUtils()
        db.insertSuitcaseInfo("1546165200","xSAX","SXAS","a")
        sno ,flightId,dst,exit  = db.querySuitcaseInfo("1546165200")
        print(sno)
        print(flightId)
        print(dst)
        print(exit)
    except Exception as E:
        print(E)