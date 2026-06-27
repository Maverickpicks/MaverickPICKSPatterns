from data_loader import load_stock

from trend_engine_v2 import trend_analysis


data=load_stock(

    "BEL"

)


result=trend_analysis(

    data["daily"]

)


print()

print(result)