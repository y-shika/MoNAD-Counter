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
	// �p���N���X��private�ϐ��g�������Ȃ�, private -> protected�ɂ���.
private:
	std::string inputFileName;
	void split(std::string& line);

public:
	// RowVector�Ɠ������Ƃ���, std::vector�ɓ���q�ɂ����ꍇ�̏�����(�傫���̎w��)�̕��@���s������������, �}�N����p���ăw�b�_�Ŏw�肵��.
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