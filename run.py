from book_extractor import Book_Extractor
import sys

args = sys.argv

if(2<= len(args)):
    img_name = str(args[1])
    book_extractor = Book_Extractor(img_name)
    book_extractor.run()
else:
    print("Argument Error")
