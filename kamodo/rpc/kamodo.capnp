@0xbfd16a03c247aaa9;


interface Kamodo {
  struct Map(Key, Value) {
    entries @0 :List(Entry);
    struct Entry {
      key @0 :Key;
      value @1 :Value;
    }
  }

  getFields @0 () -> (fields :Map(Text, Field));

  getMath @1 () -> (math :Map(Text, Function));

  evaluate @2 (expression: Expression) -> (value: Value);

  interface Value {
    # Wraps a numeric value in an RPC object.  This allows the value
    # to be used in subsequent evaluate() requests without the client
    # waiting for the evaluate() that returns the Value to finish.

    read @0 () -> (value :Literal);
    # Read back the raw numeric value.
  }

  struct Expression {
    # A numeric expression.

    union {
      literal @0 :Literal;
      # A literal numeric value.

      store @1 :Value;
      # A value that was (or, will be) returned by a previous
      # evaluate().

      parameter @2 :UInt32;
      # A parameter to the function (only valid in function bodies;
      # see defFunction).

      call :group {
        # Call a function on a list of parameters.
        function @3 :Function;
        params @4 :List(Expression);
      }
    }
  }

# Void: Void
# Boolean: Bool
# Integers: Int8, Int16, Int32, Int64
# Unsigned integers: UInt8, UInt16, UInt32, UInt64
# Floating-point: Float32, Float64
# Blobs: Text, Data
# Lists: List(T)


  struct Literal {
    union {
      void @0 :Void;
      bool @1 :Bool;
      int8 @2 :Int8;
      int16 @3 :Int16;
      int32 @4 :Int32;
      int64 @5 :Int64;
      uint8 @6 :UInt8;
      uint16 @7 :UInt16;
      uint32 @8 :UInt32;
      uint64 @9 :UInt64;
      float32 @10 :Float32;
      float64 @11 :Float64;
      text @12 :Text;
      data @13 :Data;
      list @14 :List(Literal);
      array @15 :Array;
      int @16 :Text;
      listint64 @17 :List(Int64);
      listfloat64 @18 :List(Float64);
      rational @19 :Rational;
    }
  }

  struct Rational {
    p @0 :Int64;
    q @1 :Int64;
  }


  # everything needed for registration
  struct Field {
    func @0 :Function;
    meta @1 :Meta;
    data @2 :Array;
  }

  # match kamodo's meta attribute
  struct Meta {
    units @0 :Text;
    argUnits @1 :Map(Text, Text);
    citation @2 :Text;
    equation @3 :Text; # latex expression
    hiddenArgs @4 :List(Text);
  }

  struct Argument {
    name @0 :Text;
    value @1 :Literal;
  }

  # needs to be an interface
  struct Array {
    data @0 :Data;
    shape @1 :List(UInt32);
    dtype @2 :Text;

  }

  interface Function {
    # A pythonic function f(*args, **kwargs)
    call @0 (args :List(Literal), kwargs :List(Argument)) -> (result: Literal);
    getArgs @1 () -> (args :List(Text));
    getKwargs @2 () -> (kwargs: List(Argument));
  }

}


