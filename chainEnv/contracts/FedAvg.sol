//执行FedAvg，假设本地用户的模型的参数的形式是一个长度为10的一维数组
pragma solidity ^0.4.0;

contract FedAvg{

    uint public para_index;  //这是模型参数的序号

    //构建一个模型信息的数据类型model_info
    struct model_info{
        uint256[10] para;  //模型参数para，一个长度为10的一维数组
        uint id;        //数据条的序号
        address user_address;    //该模型所来自的地址
        bool is_local;           //该模型是否为本地模型，若是False就是聚合出来的全局模型
    }

    mapping(uint => model_info) id_to_info ;//建立一个信息id到信息的映射

    //返回指定用户(msg.sender)的地址名称
    function return_Address() constant returns (address) {
        /*msg.sender*/
        return msg.sender;
    }

    //先初始化链的存储结构
   function Init() public {
       para_index = 0;
   }

    //用户向链上传模型参数的函数操作
   function add_para_to_chain(uint256[10] para) public {
       para_index += 1 ;
       bool local = true ; 
       model_info memory m_f = model_info(para, para_index, msg.sender, local) ;
       id_to_info[para_index] = m_f ; 
   }

    //查看目前链中有多少个新加入的本地模型，返回的number值是新加入的本地模型的数量
    function get_recent_local_number() view public returns (uint number){
        for (uint i = para_index; i > 0; i--){
            model_info memory tem_model_info = id_to_info[i] ; 
            bool local_identify = tem_model_info.is_local ;
            if (local_identify == false) {
                break ;
            }
        }
        number = para_index - i ; 
        return(number) ; 
    }

    //链将得到的模型聚合起来的函数操作，其中number是新加入的本地模型的数量，也就是待聚合的模型的数量
    function Avg(uint number) public {
        para_index += 1 ;
        uint[10] memory container = id_to_info[para_index].para ;
        for(uint i = para_index ; i <para_index - number ; i--){
            for(uint j = 1 ; j < 11 ; j++){
                container[j] = container[j] + id_to_info[i].para[j] ; 
            }
        }
        for(uint jj = 1 ; jj < 11 ; jj++){
            container[jj] = container[jj] / number ; 
        } 
        id_to_info[para_index].para = container ;
        id_to_info[para_index].is_local = false ;
        id_to_info[para_index].user_address = msg.sender ;
        id_to_info[para_index].id = para_index ; 
    }

    function get_ordered_model_info(uint order_number) returns (uint256[10]){
        return(id_to_info[order_number].para) ; 
    }

    //用户访问链中的聚合好的全局模型的操作
    function get_recent_global_model_info() returns (uint256[10], uint, address){
        for (uint i = para_index; i > 0; i--){
            model_info memory tem_model_info = id_to_info[i] ; 
            bool local_identify = tem_model_info.is_local ;
            if (local_identify == false) {
                break ;
            }
        }
        uint number = para_index - i ; 
        model_info recent_global = id_to_info[number] ;
        return(recent_global.para, recent_global.id, recent_global.user_address) ;
    }
}