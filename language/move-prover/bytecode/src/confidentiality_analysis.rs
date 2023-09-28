// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0

//! This confidentiality analysis flags explicit and inplicit flow of data
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt::Debug,
};

use codespan::FileId;
use codespan_reporting::diagnostic::{Diagnostic, Label, Severity};

use crate::{
    dataflow_analysis::{DataflowAnalysis, TransferFunctions},
    dataflow_domains::{AbstractDomain, JoinResult, CustomState},
    function_target::FunctionData,
    function_target_pipeline::{FunctionTargetProcessor, FunctionTargetsHolder},
    stackless_bytecode::{Bytecode, Operation},
    stackless_control_flow_graph::StacklessControlFlowGraph,
};
use move_binary_format::file_format::CodeOffset;
use move_model::{ast::TempIndex, model::FunctionEnv};

// =================================================================================================
// Data Model

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AbsValue {
    P,
    S,
}

impl AbsValue {
    pub fn is_secret(&self) -> bool {
        matches!(self, Self::S)
    }
}

type ConfidentialityAnalysisState = CustomState<TempIndex, AbsValue>;

impl ConfidentialityAnalysisState {
    fn get_pc_value(&self) -> &AbsValue {
        self.get_stack()
            .back()
            .unwrap_or_else(|| panic!("Unbound pc in state {:?}", self))
    }

    fn push_pc_value(&mut self, value: AbsValue) {
        self.get_stack_mut().push_back(value);
    }

    fn get_local_index(&self, i: &TempIndex) -> &AbsValue {
        self.get_map()
            .get(&i)
            .unwrap_or_else(|| panic!("Unbound local in state {:?}", self))
    }

    fn assign(&mut self, lhs: TempIndex, rhs: &TempIndex) {
        let rhs_value = *self.get_local_index(rhs);
        self.get_map_mut().insert(lhs, rhs_value);
    }

    fn add_local(&mut self, i: TempIndex, value: AbsValue) {
        self.get_map_mut().insert(i, value);
    }

    /// Add locals to state with value
    fn call(&mut self, locals: &Vec<TempIndex>, value: AbsValue) {
        for loc_index in locals {
            self.add_local(
                *loc_index,
                value
            );
        }
    }
}

// =================================================================================================
// Joins

impl AbstractDomain for AbsValue {
    fn join(&mut self, other: &Self) -> JoinResult {
        if self == other {
            return JoinResult::Unchanged;
        }
        // unequal; use top value
        *self = AbsValue::S;
        JoinResult::Changed
    }
}

// =================================================================================================
// Transfer functions

#[derive(PartialOrd, PartialEq, Eq, Ord)]
struct WarningId {
    index: usize,
    offset: CodeOffset,
}

struct ConfidentialityAnalysis<'a> {
    func_env: &'a FunctionEnv<'a>,
    /// Warnings about data flows to surface to the programmer
    // Uses a map instead of a vec to avoid reporting multiple warnings
    // at program locations in a loop during fixpoint iteration
    leak_warnings: RefCell<BTreeMap<WarningId, Diagnostic<FileId>>>,
}

impl ConfidentialityAnalysis<'_> {
    pub fn add_leaking_call_warning(
        &self,
        call_index: usize,
        is_explicit: bool,
        offset: CodeOffset,
    ) {
        let message = if is_explicit {
            format!("Explicit data leak via call with local {}", call_index)
        } else {
            format!("Implicit data leak via call")
        };
        let fun_loc = self.func_env.get_loc();
        let label = Label::primary(fun_loc.file_id(), fun_loc.span());
        let severity = Severity::Warning;
        let warning_id = WarningId {
            index: call_index,
            offset,
        };
        self.leak_warnings.borrow_mut().insert(
            warning_id,
            Diagnostic::new(severity)
                .with_message(message)
                .with_labels(vec![label]),
        );
    }

    pub fn add_leaking_return_warning(
        &self,
        ret_index: usize,
        is_explicit: bool,
        offset: CodeOffset,
    ) {
        let message = if is_explicit {
            format!("Explicit data leak via return of local {}", ret_index)
        } else {
            format!("Implicit data leak via return - off: {}", offset)
        };
        let fun_loc = self.func_env.get_loc();
        let label = Label::primary(fun_loc.file_id(), fun_loc.span());
        let severity = Severity::Warning;
        let warning_id = WarningId {
            index: ret_index,
            offset,
        };
        self.leak_warnings.borrow_mut().insert(
            warning_id,
            Diagnostic::new(severity)
                .with_message(message)
                .with_labels(vec![label]),
        );
    }
}

impl<'a> TransferFunctions for ConfidentialityAnalysis<'a> {
    type State = ConfidentialityAnalysisState;
    //false -> forward data flow analysis | true -> backward data flow analysis
    const BACKWARD: bool = false;

    fn execute(&self, state: &mut Self::State, instr: &Bytecode, offset: CodeOffset) {
        use Bytecode::*;
        use Operation::*;

        match instr {
            Call(_, rets, oper, args, _) => {
                // map secret args
                let secret_args: Vec<&TempIndex> = args
                    .iter()
                    .filter(|arg_index| state.get_local_index(arg_index).is_secret())
                    .collect();

                    state.call(
                        rets,
                        if secret_args.is_empty() {
                            AbsValue::P
                        } else {
                            AbsValue::S
                        }
                    );

                match oper {
                    Function(..) => {
                        // flag call during secret pc state - implicit flow
                        if state.get_pc_value().is_secret() {
                            self.add_leaking_call_warning(0, false, offset);
                        } else {
                            // check for explicit flow via call args - flag secret args
                            for secret_arg in secret_args {
                                self.add_leaking_call_warning(*secret_arg, true, offset);
                            }
                        }
                    },
                    BorrowField(..) => (),
                    BorrowGlobal(..) => (),
                    ReadRef | MoveFrom(..) | Exists(..) | Pack(..) | Eq | Neq | CastU8 | CastU64
                    | CastU128 | Not | Add | Sub | Mul | Div | Mod | BitOr | BitAnd | Xor | Shl
                    | Shr | Lt | Gt | Le | Ge | Or | And => (),
                    BorrowLoc => (),
                    Unpack(..) => (),
                    FreezeRef => (),
                    WriteRef | MoveTo(..) => (),
                    Uninit => (),
                    Destroy => (),
                    oper => panic!("unsupported oper {:?}", oper),
                }
            }
            Ret(_, rets) => {
                if !rets.is_empty() {
                    // flag returns during secret pc state - implicit flow
                    if state.get_pc_value().is_secret() {
                        self.add_leaking_return_warning(0, false, offset);
                    } else {
                        // flag returns of secret locals - explicit flow
                        for ret_index in rets {
                            if state.get_local_index(ret_index).is_secret() {
                                self.add_leaking_return_warning(*ret_index, true, offset);
                            }
                        }
                    }
                }
            }
            Branch(_, _, _, guard) => {
                state.push_pc_value(*state.get_local_index(guard))
            }
            Assign(_, lhs, rhs, _) => {
                state.assign(*lhs, rhs);
            }
            Load(_, lhs, _) => {
                state.add_local(*lhs, *state.get_pc_value());
            }
            Jump(..) | Label(..) | Abort(..) | Nop(..) | SaveMem(..) | SaveSpecVar(..)
            | Prop(..) => (),
        }
    }
}

impl<'a> DataflowAnalysis for ConfidentialityAnalysis<'a> {}
pub struct ConfidentialityAnalysisProcessor();
impl ConfidentialityAnalysisProcessor {
    pub fn new() -> Box<Self> {
        Box::new(ConfidentialityAnalysisProcessor())
    }
}

impl FunctionTargetProcessor for ConfidentialityAnalysisProcessor {
    fn process(
        &self,
        _targets: &mut FunctionTargetsHolder,
        func_env: &FunctionEnv,
        data: FunctionData,
        _scc_opt: Option<&[FunctionEnv]>,
    ) -> FunctionData {
        if func_env.is_native() {
            return data;
        }

        let mut initial_state = ConfidentialityAnalysisState::default();
        // map every local
        for tmp_idx in 0..func_env.get_local_count() {
            // if local is a parameter, flag it as secret
            // TODO: need to considerate invariants
            initial_state.add_local(
                tmp_idx,
                if tmp_idx < func_env.get_parameter_count() {
                    AbsValue::S
                } else {
                    AbsValue::P
                },
            );
        }

        // initialize pc value
        initial_state.push_pc_value(AbsValue::P);

        let cfg = StacklessControlFlowGraph::new_forward(&data.code);
        let analysis = ConfidentialityAnalysis {
            func_env,
            leak_warnings: RefCell::new(BTreeMap::new()),
        };
        analysis.analyze_function(initial_state, &data.code, &cfg);
        let env = func_env.module_env.env;
        for (_, warning) in analysis.leak_warnings.into_inner() {
            env.add_diag(warning)
        }
        data
    }

    fn name(&self) -> String {
        "confidentiality_analysis".to_string()
    }
}
