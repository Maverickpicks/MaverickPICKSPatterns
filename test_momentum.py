from data_loader import load_stock

from momentum_engine import momentum_analysis


data=load_stock(

    "BEL"

)


daily=data["daily"]


result=momentum_analysis(

    daily

)


print()

print(result)