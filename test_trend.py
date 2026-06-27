from data_loader import load_stock

from trend_engine import trend_analysis


data=load_stock(

    "BEL"

)


daily=data["daily"]


result=trend_analysis(

    daily

)


print()

print(result)