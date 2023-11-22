//UNLICENSE
pragma solidity ^0.8.0;

contract NetworkManager{
    // Info represent a federated training.
    constructor() {
        owner = payable(msg.sender);
    }
    struct FLMeta{
        uint client_num;
        uint training_round;
        string fl_algorithm;
    }
    // Record Training Result for client, this struct simulate a float
    // result = a * 10^b
    struct TrainResult{
        uint a;
        uint b;
        uint epoch;
    }

    address public owner;
    mapping(address => TrainResult) regist_result;
    address[] public regist_client;
    //mapping(address => TrainResult) public regist_client;

    FLMeta fl_info;
    TrainResult[] fl_result;
    
    modifier only_owner{
        require(msg.sender == owner,"Only Owner Can Call This Function");
        _;
    }
    modifier only_participant{
        require(is_duplicate_client(msg.sender), "Onlu participant can call this func");
        _;
    }
    function fl_init(uint client_num, uint round, string calldata fl_algorithm) public only_owner{
        fl_info.client_num = client_num;
        fl_info.training_round = round;
        fl_info.fl_algorithm = fl_algorithm;
    }
    function is_duplicate_client(address client) internal view returns(bool){
        for(uint i = 0; i < regist_client.length; i++){
            if(regist_client[i] == client) return true;
        }
        return false;
    }
    function client_regist() public {
        require(!is_duplicate_client(msg.sender),"Client has registed!");
        regist_client.push(msg.sender);
        regist_result[msg.sender] = TrainResult(0,0,0);
    }

    function upload_result(uint a, uint b, uint epoch) public only_participant{
        TrainResult memory tmp;
        tmp.a = a;
        tmp.b = b;
        tmp.epoch = epoch;
        regist_result[msg.sender] = tmp;
    }


}