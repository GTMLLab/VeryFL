pragma solidity >=0.4.22;

contract SimpleStorage{
    uint storageData;

    function set(uint x) public{
        storageData = x;
    }

    function get(uint x) public view returns (uint){
        return storageData;
    }
}
