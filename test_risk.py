from data_loader import load_stock

from risk_engine import risk_analysis


data=load_stock(

    "BEL"

)


result=risk_analysis(

    data["daily"]

)


print()

print(result)