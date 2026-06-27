from data_loader import load_stock

from trend_engine import trend_analysis

from momentum_engine import momentum_analysis

from relative_strength import relative_strength_analysis

from volume_engine import volume_analysis

from risk_engine import risk_analysis

from pattern_engine import pattern_analysis

from ranking_engine import ranking_engine

from data_loader import load_nifty


data=load_stock(

    "BEL"

)


nifty=load_nifty()


trend=trend_analysis(

    data["daily"]

)


momentum=momentum_analysis(

    data["daily"]

)


rs=relative_strength_analysis(

    data["daily"],

    nifty

)


volume=volume_analysis(

    data["daily"],

    data["weekly"],

    data["monthly"]

)


risk=risk_analysis(

    data["daily"]

)


pattern=pattern_analysis(

    data["daily"]

)


result=ranking_engine(

    trend,

    momentum,

    rs,

    volume,

    pattern,

    risk

)


print()

print("TREND")

print(trend)


print()

print("MOMENTUM")

print(momentum)


print()

print("RS")

print(rs)


print()

print("VOLUME")

print(volume)


print()

print("PATTERN")

print(pattern)


print()

print("RISK")

print(risk)


print()

print("FINAL")

print(result)