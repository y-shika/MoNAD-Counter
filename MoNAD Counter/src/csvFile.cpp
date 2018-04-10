#include "csvFile.h"

csvFile::csvFile() {
	// 相対パスで指定している. "data"フォルダにcsvファイルを置いておく.
	inputFileName = "data\\counter_data.csv";
}

void csvFile::read() {
	std::ifstream inputFile(inputFileName);
	if (inputFile.fail()) {
		std::cout << "ファイルを開けません.\n";

		// キー入力待ち
		getchar();

		exit(0);
	}
	else std::cout << "ファイルを開きました.\n";

	std::string line;

	getline(inputFile, line);

	// 入力を一行ごとに分割
	while (getline(inputFile, line)) split(line);
	inputFile.close();
}

// 行中での分割
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