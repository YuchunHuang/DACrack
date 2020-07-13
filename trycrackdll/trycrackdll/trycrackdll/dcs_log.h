#pragma once
#include "dcs_logIf.h"
#include <mutex>

using namespace std;

class DCS_LOG_API CDcd_log:public IDCS_LOG
{
public:
	CDcd_log();
	~CDcd_log();
public:
	//获取实例对象
    IDCS_LOG* GetInstance();

	//初始化
    bool InitLogger(const char* file_name, int min_level);
	//写日志
    void WriteLog(int level, const char* exec_file, int exec_line, int tid, const char* format, ...);
    //无锁写日志
	void WriteLogNoLock(int level, const char* exec_file, int exec_line, int tid, const char* format, ...);

	string GetLogPath();
private:
    void set_log(int level, const char* exec_file, int exec_line, int tid, const char* format, va_list valst);
    
public:
    static CDcd_log*   instance_;

private:
    int     min_level_;
    char*   log_file_;
    char*   log_buf_;
    FILE*   log_fp_;
	mutex   m_mutex;
};

