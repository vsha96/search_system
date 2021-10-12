## Описание
В данном отчете подробно разобраны шаги для создания примитивной поисковой системы, использующей в своей основе векторное представление документов, где значения рассчитываются через tf_idf (двумя способами подсчета tf: count и log(1+count))  

Здесь документы - это предложения из коллекции текстов  
Тексты взяты из прикрепленных ссылок к интересным фактам Википедии  
Запросы - сами формулировку этих фактов (можно увидеть в частях **Запросы** и **Поиск по запросу**)  

При поиске по запросу, сам запрос тоже переводится в векторное пространство документов, и система выдает релевантные документы по мере их близости (близость считается как cos между векторами)  

## Подключение модуля
```import mysearchsys as mss```

## Процедуры и функции
Процедура сборки коллекции, если добавили файлы в text  
```mss.make_collection()```  

Функция поиска по коллекции, выдает кортеж:  
1) список близких документов по мере tf-idf (tf = count)  
2) список по мере tf-idf (tf = log(1+count))  
Пример: ```mss.search('На складе чугуноплавильного завода скопился годовой запас продукции')```  

Пример, как различать разные подсчеты tf: 
```
# список релевантных документов для запроса
#   если первая индексация [0], тогда tf = count
#   если первая индексация [1], тогда tf = log(1+count)
search_list = mss.search('Внешность короля Генриха III')[0][:3]

print(search_list)
```
  
  
## Некоторые ключевые моменты и удобные возможности
- Даты, римские цифры, английские названия - тоже термы (это может играть роль при поиске документа)  
- Есть возможность сохранять или пересобирать коллекцию с помощью make_collection()  
- Автоматическое добавление новых текстов в коллекцию (нужно записывать новые факты в виде fact_*\<number>*.txt в директорию text и перезапустить сборку коллекции)  
- Обработанная коллекция сохраняется и подгружается как объект pickle (в директории obj)  