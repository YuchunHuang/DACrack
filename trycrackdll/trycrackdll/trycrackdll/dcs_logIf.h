#pragma once

#ifdef DCS_LOG_EXPORTS
#define DCS_LOG_API __declspec(dllexport)
#else
#define DCS_LOG_API __declspec(dllimport)
#endif

#ifndef DCS_LOG_EXPORTS
#pragma comment(lib, "dcs_log.lib")
#endif

#define gettid() 0

#define LEVEL_SOCKET_DEBUG  0
#define LEVEL_DEBUG         1
#define LEVEL_INFO          2
#define LEVEL_WARNING       3
#define LEVEL_ERROR         4

#define LOG_BUF_SIZE        2048
#define MAX_FILENAME_LEN    256

using namespace std;

class IDCS_LOG;
extern "C" DCS_LOG_API IDCS_LOG * g_pSyslog;
#define LOG_INIT(log_name, min_level) \
	if(g_pSyslog)\
		g_pSyslog->InitLogger(log_name, min_level)

#define LOG_INFO_NOLOCK(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLogNoLock(LEVEL_INFO, \
                                        __FILE__, __LINE__, gettid(), \
                                        format, ##__VA_ARGS__)

#define SOCKET_DEBUG(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLog(LEVEL_SOCKET_DEBUG, \
                                        __FILE__, __LINE__, gettid(), \
                                        format, ##__VA_ARGS__)

#define LOG_DEBUG(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLog(LEVEL_DEBUG, \
                                        __FILE__, __LINE__, gettid(), \
                                        format, ##__VA_ARGS__)

#define LOG_INFO(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLog(LEVEL_INFO, \
                                        __FILE__, __LINE__, gettid(), \
                                        format, ##__VA_ARGS__)

#define LOG_WARNING(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLog(LEVEL_WARNING, \
                                        __FILE__, __LINE__, gettid(), \
                                        format, ##__VA_ARGS__)

#define LOG_ERROR(format, ...) \
	if(g_pSyslog)\
		g_pSyslog->WriteLog(LEVEL_ERROR, __FILE__, __LINE__, gettid(), format, ##__VA_ARGS__)

class IDCS_LOG 
{
public:
	//获取实例对象
	virtual IDCS_LOG* GetInstance()= 0;

	//初始化
	virtual bool InitLogger(const char* file_name, int min_level)=0;
	//写日志
	virtual void WriteLog(int level, const char* exec_file, int exec_line, int tid, const char* format, ...)=0;
	//无锁写日志
	virtual void WriteLogNoLock(int level, const char* exec_file, int exec_line, int tid, const char* format, ...)=0;

};
