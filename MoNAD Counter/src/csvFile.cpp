#include "csvFile.h"

csvFile::csvFile() {
	// ���΃p�X�Ŏw�肵�Ă���. "data"�t�H���_��csv�t�@�C����u���Ă���.
	inputFileName = "data\\counter_data.csv";
}

void csvFile::read() {
	std::ifstream inputFile(inputFileName);
	if (inputFile.fail()) {
		std::cout << "�t�@�C�����J���܂���.\n";

		// �L�[���͑҂�
		getchar();

		exit(0);
	}
	else std::cout << "�t�@�C�����J���܂���.\n";

	std::string line;

	getline(inputFile, line);

	// ���͂���s���Ƃɕ���
	while (getline(inputFile, line)) split(line);
	inputFile.close();
}

// �s���ł̕���
void csvFile::split(std::string& line) {
	std::istringstream lineStream(line);

	int inoutCount = 0;
	Eigen::Matrix<double, 1, INNUM> inData;
	Eigen::Matrix<double, 1, OUTNUM> outData;
	Eigen::Matrix<double, 1, RLNUM> rlData;
	Eigen::Matrix<double, 1, SSUNUM> ssuData;
	Eigen::Matrix<double, 1, BLNUM> blData;
	std::string field;

	for (int i = 0; i < INNUM; i++) {
		getline(lineStream, field, ',');
		inData[i] = std::stod(field);
	}
	Teach_in.push_back(inData);

	for (int i = 0; i < OUTNUM; i++) {
		getline(lineStream, field, ',');
		outData[i] = std::stod(field);
	}
	Teach_out.push_back(outData);

	for (int i = 0; i < RLNUM; i++) {
		getline(lineStream, field, ',');
		rlData[i] = std::stod(field);
	}
	Teach_rl.push_back(rlData);

	for (int i = 0; i < SSUNUM; i++) {
		getline(lineStream, field, ',');
		ssuData[i] = std::stod(field);
	}
	Teach_ssu.push_back(ssuData);

	for (int i = 0; i < BLNUM; i++) {
		getline(lineStream, field, ',');
		blData[i] = std::stod(field);
	}
	Teach_bl.push_back(blData);
}