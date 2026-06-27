from data_loader import load_stock

from momentum_engine_v2 import momentum_analysis


data=load_stock(

    "BEL"

)


result=momentum_analysis(

    data["daily"]

)


print()

print(result)