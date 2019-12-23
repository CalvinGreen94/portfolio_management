import matplotlib.pyplot as plt
target = int(float(input('please enter the target profit $')))
print('-proft target:${}'.format(target))
# starting_balance = 50000.00
# current_balance = 50000.00
# sim_balance = 51000.00
starting_balance = int(float(input('please enter the balance your account started with $')))
current_balance = int(float(input('please enter your current balance $')))
print('-current balance: ${}'.format(current_balance))
loss = starting_balance - current_balance
# sim_loss = starting_balance-sim_balance
print('-loss: ${}'.format(loss))
#total_contracts = 3
total_contracts = int(input('please enter the amount of contracts you would like to trade'))
print('-total_contracts:',total_contracts)
# contracts = {'cl':54.06,
#               'es':2980.00,
#               'rty':1526.90}
# print('-contracts:',contracts)
# contracts_price_10 = 54.06
# print('-contracts_price:',contracts_price_10)
max_risk = current_balance*.01/3
print('----max risk: ${}'.format(max_risk))
contracts_tick_value_50 = 50.00 *total_contracts
contracts_tick_value_12 = 12.50 * total_contracts
contracts_tick_value_10 = 10.00 * total_contracts
contracts_tick_value_5 = 5.00 *total_contracts
contracts_tick_value_1 = 1.00 * total_contracts
print('-contracts_tick_value 50:${:.2f}'.format(contracts_tick_value_50))
print('-contracts_tick_value 12:${:.2f}'.format(contracts_tick_value_12))
print('-contracts_tick_value 10:${:.2f}'.format(contracts_tick_value_10))
print('-contracts_tick_value 5:${:.2f}'.format(contracts_tick_value_5))
print('-contracts_tick_value 1:${:.2f}'.format(contracts_tick_value_10))

if current_balance < starting_balance:
    print('-current balance is Less than starting balance .. continuing to perform risk analysis to reach target profit')
    current_balance = starting_balance-loss
    print('-current_balance: ${:.2f}'.format(current_balance))
    current_target_loss = target - current_balance
    print('-current loss from target profit: ${:.2f}'.format(current_target_loss))
    daily_target = current_target_loss/12
    print('-daily target: ${:.2f}'.format(daily_target))
    hourly_target = daily_target/4
    print('-hourly target: ${:.2f}'.format(hourly_target))
#     for contract in contracts:
    print('-Target contract size in each market to reach daily target')
    profit_size50 = daily_target/contracts_tick_value_50
    print('-profit size for $50 tick size: ',profit_size50)
    max_loss50 = profit_size50 *.015
    print('max loss: $',max_loss50*daily_target)
    profit_size12 = daily_target/contracts_tick_value_12
    print('-profit size for $12 tick size:',profit_size12)
    max_loss12 = profit_size12 *.015
    print('max loss: $',max_loss12*daily_target)
    profit_size10 = daily_target/contracts_tick_value_10
    print('-profit size for $10 tick size:',profit_size10)
    max_loss10 = profit_size10 *.015
    print('max loss: $',max_loss10*daily_target)
    profit_size5 = daily_target/contracts_tick_value_5
    print('-profit size for $5 tick size: ',profit_size5)
    max_loss5 = profit_size5 *.015
    print('max loss: $',max_loss5*daily_target)
    profit_size1 = daily_target/contracts_tick_value_1
    print('-profit size for $1 tick size:',profit_size1)
    max_loss1 = profit_size1 *.015
    print('max loss: $',max_loss1*daily_target)
elif current_balance == starting_balance:
    print('-current balance is Less than starting balance .. continuing to perform risk analysis to reach target profit')
    current_balance = starting_balance-loss
    print('-current_balance: ${:.2f}'.format(current_balance))
    current_target_loss = target - current_balance
    print('-current loss from target profit: ${:.2f}'.format(current_target_loss))
    daily_target = current_target_loss/12
    print('-daily target: ${:.2f}'.format(daily_target))
    hourly_target = daily_target/4
    print('-hourly target: ${:.2f}'.format(hourly_target))
#     for contract in contracts:
    print('-Target contract size in each market to reach daily target')
    profit_size50 = daily_target/contracts_tick_value_50
    print('-profit size for $50 tick size: ',profit_size50)
    max_loss50 = profit_size50 *.015
    print('max loss: $',max_loss50*daily_target)
    profit_size12 = daily_target/contracts_tick_value_12
    print('-profit size for $12 tick size:',profit_size12)
    max_loss12 = profit_size12 *.015
    print('max loss: $',max_loss12*daily_target)
    profit_size10 = daily_target/contracts_tick_value_10
    print('-profit size for $10 tick size:',profit_size10)
    max_loss10 = profit_size10 *.015
    print('max loss: $',max_loss10*daily_target)
    profit_size5 = daily_target/contracts_tick_value_5
    print('-profit size for $5 tick size: ',profit_size5)
    max_loss5 = profit_size5 *.015
    print('max loss: $',max_loss5*daily_target)
    profit_size1 = daily_target/contracts_tick_value_1
    print('-profit size for $1 tick size:',profit_size1)
    max_loss1 = profit_size1 *.015
    print('max loss: $',max_loss1*daily_target)

elif current_balance >= starting_balance:
    print('-current balance is Less than starting balance .. continuing to perform risk analysis to reach target profit')
    current_balance = starting_balance-loss
    print('-current_balance: ${:.2f}'.format(current_balance))
    current_target_loss = target - current_balance
    print('-current loss from target profit: ${:.2f}'.format(current_target_loss))
    daily_target = current_target_loss/12
    print('-daily target: ${:.2f}'.format(daily_target))
    hourly_target = daily_target/4
    print('-hourly target: ${:.2f}'.format(hourly_target))
#     for contract in contracts:
    print('-Target contract size in each market to reach daily target')
    profit_size50 = daily_target/contracts_tick_value_50
    print('-profit size for $50 tick size: ',profit_size50)
    max_loss50 = profit_size50 *.015
    print('max loss: $',max_loss50*daily_target)
    profit_size12 = daily_target/contracts_tick_value_12
    print('-profit size for $12 tick size:',profit_size12)
    max_loss12 = profit_size12 *.015
    print('max loss: $',max_loss12*daily_target)
    profit_size10 = daily_target/contracts_tick_value_10
    print('-profit size for $10 tick size:',profit_size10)
    max_loss10 = profit_size10 *.015
    print('max loss: $',max_loss10*daily_target)
    profit_size5 = daily_target/contracts_tick_value_5
    print('-profit size for $5 tick size: ',profit_size5)
    max_loss5 = profit_size5 *.015
    print('max loss: $',max_loss5*daily_target)
    profit_size1 = daily_target/contracts_tick_value_1
    print('-profit size for $1 tick size:',profit_size1)
    max_loss1 = profit_size1 *.015
    print('max loss: $',max_loss1*daily_target)
