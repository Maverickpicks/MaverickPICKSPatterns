from data_loader import load_stock

from pattern_engine import pattern_analysis


data=load_stock(

    "BEL"

)


result=pattern_analysis(

    data["daily"]

)


print()

print(result)