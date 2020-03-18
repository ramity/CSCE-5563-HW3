import csv
import Levenshtein as Lev
import sys
import numpy as np

def main():
	student_preds, gold_preds=[],[]
	student_file=""
	try:
		student_file=sys.argv[1]
	except IndexError:
		print("Error! Please add your filename while running the code")
		exit()

	gold_file="gold_labels.csv"
	with open(student_file) as f:
		rows_student=csv.reader(f)
		for row in rows_student:
			student_preds.append(row[1])
	with open(gold_file) as f:
		rows_gold=csv.reader(f)
		for row in rows_gold:
			gold_preds.append(row[1])
	score=[]
	if len(student_preds)!=len(gold_preds):
		print("Warning: Number of rows inconsistent!!")
	for rs, rg in zip(student_preds, gold_preds):
		distance = Lev.distance(rs,rg)
		print(rs)
		print(rg)
		print(distance)
		print('-'*40)
		score.append(distance)
	average_distance=np.mean(score)
	print("\n")
	print('-'*40)
	print('Average levenshtein distance: {:1.2f}'.format(average_distance))
	print('-'*40)
	print("\n")
main()
