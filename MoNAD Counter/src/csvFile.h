#ifndef __CSVFILE_H_INCLUDED__
#define __CSVFILE_H_INCLUDED__

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define INNUM 1
#define SSUNUM 2
#define BLNUM 2
#define HIDNUM 8
#define OUTNUM 2
#define RLNUM 2

class csvFile
{
	// 継承クラスでprivate変数使いたいなら, private -> protectedにする.
private:
	std::string inputFileName;
	void split(std::string& line);

public:
	// RowVectorと同じことだが, std::vectorに入れ子にした場合の初期化(大きさの指定)の方法が不明だったため, マクロを用いてヘッダで指定した.
	std::vector<Eigen::Matrix<double, 1, INNUM>> Teach_in;
	std::vector<Eigen::Matrix<double, 1, OUTNUM>> Teach_out;
	std::vector<Eigen::Matrix<double, 1, RLNUM>> Teach_rl;
	std::vector<Eigen::Matrix<double, 1, SSUNUM>> Teach_ssu;
	std::vector<Eigen::Matrix<double, 1, BLNUM>> Teach_bl;

	csvFile();
	//~csvFile();

	void read();
};

#endif