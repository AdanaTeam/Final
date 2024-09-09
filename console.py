def show_commands():
   print ('exit - закрытие программы\n')



def run(command_list):


    print('Adana v0.0: Нейронная сеть, анализирующая изображения на следы редактирования')
    print('https://github.com/AdanaTeam/Final')
    print('Для получения списка команд введите "help"\n')
    finished = False
    while (not finished):
    
        command=input('Введите команду: ')
        if command in command_list.keys():
            command_list[command]()
        else:
            print(f"Команда  \"{command}\" не найдена.\n")

# run(command_list)