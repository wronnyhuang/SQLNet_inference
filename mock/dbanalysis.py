import csv, sys, sqlite3

def create_connection(db_file):
	con = sqlite3.connect(db_file)
	return con

#load the table
def load_data(con,tname,fpath):
	cur = con.cursor()
	q1='drop table if exists {}'.format(tname) 
	cur.execute(q1)
	q2='''CREATE TABLE {} (account_number,age,gender,segment,accommodation_spending,food_spending,air_transportation_spending,gasoline_spending,information_technology_spending,utilities_spending,days_account_open,annual_income)'''.format(tname)
	cur.execute(q2) # use your column names here

	with open(fpath,'rb') as fin:
    # csv.DictReader uses first line in file for column headings by default
    		dr = csv.DictReader(fin) # comma is default delimiter
    		to_db = [(i['account_number'], i['age'], i['gender'], i['segment'],i['accommodation_spending'],i['food_spending'],i['air_transportation_spending'],i['gasoline_spending'],i['information_technology_spending'],i['utilities_spending'],i['days_account_open'],i['annual_income']) for i in dr]

		q3='''INSERT INTO {} (account_number,age,gender,segment,accommodation_spending,food_spending,air_transportation_spending,gasoline_spending,information_technology_spending,utilities_spending,days_account_open,annual_income) VALUES (?, ?,?,?,?,?,?,?,?,?,?,?)'''.format(tname)
		cur.executemany(q3, to_db)
		con.commit()

#fetch rows
def select_row(con,uquery):
	cur = con.cursor()
	cur.execute(uquery)
	rows = cur.fetchall()
	for row in rows:
        	print(row)

def main():
	db_name="gkumar.db"
	tname=sys.argv[1]
	#fpath="/Users/na974cq/downloads/Personal/acct_test.csv"
	uquery=sys.argv[2]
	fpath=sys.argv[3]
	#uquery="select account_number from acct where age<='28' AND annual_income<='80000'"
	con=create_connection(db_name)
	with con:
 		print("1.create the table and load the data")
		load_data(con,tname,fpath)
		
		print("2.Fetch data based on user query")
		select_row(con,uquery)
if __name__ == "__main__":
	main()
