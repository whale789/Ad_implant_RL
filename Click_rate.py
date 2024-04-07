import matplotlib.pyplot as plt
#
# file_path="Datas/click data.txt"
# definitely_will=0
# maybe=0
# probably_not=0
# certainly_not=0
# with open(file_path, newline='',encoding='utf-8', errors='ignore') as file:
#     for line in file:
#         line = line.strip()
#         click_data = line.split(',')
#         for i in range(len(click_data)):
#             if click_data[i]=="肯定会":
#                 definitely_will+=1
#             elif click_data[i]=="可能会":
#                 maybe+=1
#             elif click_data[i]=="可能不会":
#                 probably_not+=1
#             elif click_data[i]=="肯定不会":
#                 certainly_not+=1
# # print(definitely_will,maybe,probably_not,certainly_not)
#
#
# # 数据
# sizes = [definitely_will, maybe, probably_not, certainly_not]
# labels = ['肯定会', '可能会', '可能不会', '肯定不会']
#
# # 创建扇形图
# plt.figure(figsize=(10, 10),dpi=110)
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
#
# # 添加图例
# plt.legend(labels, loc=(0.95,0.7))
#
# # 标题
# plt.title('点击率')
#
# # 关闭坐标轴
# # plt.axis('equal')
# plt.axis('off')
#
# # 显示图表
# plt.show()

def click_count(index):
    file_path="Datas/click data.txt"
    click=0
    not_click=0
    with open(file_path, newline='',encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            click_data = line.split(',')
            # for i in range(len(click_data)):
            if click_data[index]=="肯定会" or click_data[index]=="可能会":
                click+=1
            elif click_data[index]=="可能不会" or click_data[index]=="肯定不会":
                not_click+=1
    print(click,not_click)
    return click,not_click

ad_id=[1,2,3,4,5,6,7]
click_list=[]
not_click_list=[]
ad_list=["ad1","ad2","ad3","ad4",'ad5','ad6','ad7']
for i in range(len(ad_id)):
    click,not_click=click_count(ad_id[i])
    click_list.append(click)
    not_click_list.append(not_click)

categories = ad_list
values1 = click_list  # 第一组数据
values2 = not_click_list  # 第二组数据

# 柱形宽度
bar_width = 0.35

# 创建图形
fig, ax = plt.subplots()

# 绘制第一组柱形图
bar1 = ax.bar(range(len(categories)), values1, bar_width, label='会点击')

# 绘制第二组柱形图
bar2 = ax.bar([x + bar_width for x in range(len(categories))], values2, bar_width, label='不会点击')

# 添加标签、标题和图例
# ax.set_xlabel('Advertisement serial number')
# ax.set_ylabel('Number of clicks')
ax.set_title("各广告点击数量")
ax.set_xlabel('广告序号')
ax.set_ylabel('是/否点击数量')
ax.set_xticks([x + bar_width / 2 for x in range(len(categories))])
ax.set_xticklabels(labels=categories)
ax.legend()
# 显示图表
plt.show()