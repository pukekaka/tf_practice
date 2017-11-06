import time
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.cluster import KMeans

user_product_dic = {}
product_user_dic = {}
product_id_name_dic = {}

for line in open('Online_Retail.txt'):
    line_items = line.strip().split('\t')
    user_code = line_items[6]
    product_id = line_items[1]
    product_name = line_items[2]

    if len(user_code) == 0:
        continue

    country = line_items[7]
    if country != 'United Kingdom':
        continue

    try:
        invoice_year = time.strptime(line_items[4], '%Y-%m-%d %H:%M').tm_year
        #2010-12-01 8:34

    except ValueError:
        continue

    #print(invoice_year)

    if invoice_year != 2011:
        continue


    user_product_dic.setdefault(user_code, set())
    user_product_dic[user_code].add(product_id)

    #print(user_product_dic)

    product_user_dic.setdefault(product_id, set())
    product_user_dic[product_id].add(user_code)

    product_id_name_dic[product_id] = product_name

product_per_user_li = [len(x) for x in user_product_dic.values()]



print('# of users:', len(user_product_dic))
print('# of products:', len(product_user_dic))

print(stats.describe(product_per_user_li))
#user_code = 1
#user_product_dic.setdefault(user_code, set())

#print(user_product_dic)

plot_data_all = Counter(product_per_user_li)
plot_data_x = list(plot_data_all.keys())
plot_data_y = list(plot_data_all.values())
plt.xlabel('고유 상품 가짓수')
plt.ylabel('사용자 수')
plt.scatter(plot_data_x, plot_data_y, marker='+')

# plt.show()
# print(product_per_user_li)

min_product_user_li = [k for k,v in user_product_dic.items() if len(v) == 1]

max_product_user_li = [k for k,v in user_product_dic.items() if len(v)>=600]

print("# of users purchased one product:%d" %(len(min_product_user_li)))
print("# of users purchased more than 600 product:%d" %(len(max_product_user_li)))

user_product_dic = {k:v for k,v in user_product_dic.items() if len(v)> 1 and len(v)<=600}
print("# of left user:%d" %(len(user_product_dic)))

id_product_dic = {}

for product_set_li in user_product_dic.values():
    for x in product_set_li:
        if x in id_product_dic:
            product_id = id_product_dic[x]

        else:
            id_product_dic.setdefault(x, len(id_product_dic))
print("# of left items:%d" %(len(id_product_dic)))

id_user_dic = {}
user_product_vec_li = []
all_product_count = len(id_product_dic)

for user_code, product_per_user_set in user_product_dic.items():
    user_product_vec = [0] * all_product_count
    id_user_dic[len(id_user_dic)] = user_code
    for product_name in product_per_user_set:
        user_product_vec[id_product_dic[product_name]] = 1

    user_product_vec_li.append(user_product_vec)

print(id_user_dic[0])
print(user_product_dic['15150'])
print(user_product_vec_li[0])
print(len(user_product_vec_li[0]))

random.shuffle(user_product_vec_li)

train_data = user_product_vec_li[:2500]
test_data = user_product_vec_li[2500:]
print("# of train data:%d, # of test_data: %d" %(len(train_data), len(test_data)))

km_predict = KMeans(n_clusters=4, n_init=10, max_iter=20).fit(train_data)
km_predict_result = km_predict.predict(test_data)
print(km_predict_result)